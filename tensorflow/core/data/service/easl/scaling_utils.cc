//
// Created by aymond on 26.11.21.
//

#include "tensorflow/core/data/service/easl/scaling_utils.h"

#include <queue>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/data/service/easl/cache_model.h"


namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace scaling_utils {

namespace {
// Represents an offset which is subtracted from the non-rounded up worker count
// This offset tries to avoid cases where a value such as 4.02 provisions
// 5 workers and not 4, as woul be ideal
double worker_count_alpha_ = 0.1;
int MAX_WORKERS_PER_JOB = 100;

double kMinBatchTimeRelativeImprovement = 0.05; // 5%
uint32 kInStabilityBeforeScaling = 20;
}


Status DetermineElasticity(
    const std::string& job_type,
    const experimental::DispatcherConfig& dispatcher_config,
    const ::tensorflow::data::easl::MetadataStore& metadata_store,
    const std::string& dataset_key,
    const int64 available_workers,
    int64& worker_count) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;
  using JobMetrics = ::tensorflow::data::easl::JobMetrics;

  // Give out max number of workers
  if(dispatcher_config.scaling_policy() == 2){
    //worker_count = available_workers;
    worker_count = MAX_WORKERS_PER_JOB;
    return Status::OK();
  }

  if(dispatcher_config.scaling_policy() == 3){
    worker_count = 1;
    return Status::OK();
  }

  std::shared_ptr<JobMetrics> job_metrics;
  Status s = metadata_store.GetJobMetricsByDatasetKey(dataset_key, job_metrics);

  // We do not yet have the metrics for this dataset --> use 1 worker
  if(errors::IsNotFound(s)) {
    VLOG(0) << "(DetermineElasticity) No metrics found for dataset "
            << dataset_key << ". Will use 1 worker in " << job_type << " mode.";
    worker_count = 1;
    return Status::OK();
  } else if (!s.ok()) {
    VLOG(0) << "(DetermineElasticity) Another error has been thrown: " << s;
    return s;
  }

//  job_metrics->model_metrics_->metrics_history_.back();
  worker_count = job_metrics->target_worker_count_;

  if (worker_count < 1){
    VLOG(0) << "(DetermineElasticity) - Target worker count not set for previous job with same dataset key.";
    worker_count = 1;
  }

  VLOG(0) << "(DetermineElasticity) Metrics found for dataset "
          << dataset_key << ". Will use " << worker_count << " workers in "
          << job_type << " mode.";

  return Status::OK();
}


Status DynamicWorkerCountUpdate(
    const std::string& job_type,
    const int64 job_id,
    const experimental::DispatcherConfig& dispatcher_config,
    ::tensorflow::data::easl::MetadataStore& metadata_store,
    int64& worker_count) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Give out max number of workers
  if(dispatcher_config.scaling_policy() == 2){
    //worker_count = available_workers;
    worker_count = MAX_WORKERS_PER_JOB;
    return Status::OK();
  }
  if(dispatcher_config.scaling_policy() == 3){
    // Alternate between 1 and 2 for testing.

    static int counter = 0;
    counter++;
    if ( counter > 20){
      worker_count = 2;
      counter = 0;
    } else if( counter > 40 ){
      worker_count = 1;
      counter = 0;
    } else {
      worker_count = 1;
    }
    return Status::OK();
  }

  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Entering.";

  bool is_scaling;
  TF_RETURN_IF_ERROR(metadata_store.IsJobScaling(job_id, is_scaling));

  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;
  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Worker count for last metrics: " <<
  metrics_history[metrics_history.size()-1]->worker_count(); // Guaranteed to succeed.

  if(is_scaling) {
    VLOG(0) << "EASL (DynamicWorkerCountUpdate) - is_scaline is true";
    if (metrics_history.size() == 1){ // Cannot be smaller than 1
      VLOG(0) << "EASL (DynamicWorkerCountUpdate) - no metrics_history -> increasing worker count";
      worker_count = metrics_history.back()->worker_count() + 1;
      metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
      return Status::OK();
    }

    std::shared_ptr<ModelMetrics::Metrics> second_to_last_metrics = metrics_history[metrics_history.size() - 2];
    std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];

    int second_to_last_index = metrics_history.size() - 2;
    while(second_to_last_metrics->worker_count() == last_metrics->worker_count()){
      if (second_to_last_index == 0){
        VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Should not enter here!"
        << "This might lead to an infinite loop! ";
        worker_count = metrics_history.back()->worker_count();
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        return Status::OK();
      }
      second_to_last_metrics = metrics_history[--second_to_last_index];
    }

    double stl_batch_time = second_to_last_metrics->last_x_batch_time_ms();
    double l_batch_time = last_metrics->last_x_batch_time_ms();
    double relative_improvement = 1.0 - l_batch_time / stl_batch_time;

    if (second_to_last_metrics->worker_count() < last_metrics->worker_count()) {
      // We are scaling up
      if (relative_improvement > kMinBatchTimeRelativeImprovement){
        worker_count = last_metrics->worker_count() + 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                     << "Improvement large enough:\n"
                     << " > improvement: " << relative_improvement << "\n"
                     << " > next worker count: " << worker_count;
      } else {
        worker_count = second_to_last_metrics->worker_count();
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                << "Improvement NOT large enough:\n"
                << " > improvement: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      }
    } else {
      // We are scaling down
      if (relative_improvement > -kMinBatchTimeRelativeImprovement) {
        worker_count = last_metrics->worker_count() - 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss ok:\n"
                << " > improvement: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      } else {
        worker_count = second_to_last_metrics->worker_count();
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss NOT ok:\n"
                << " > improvement: " << relative_improvement << "\n"
                << " > next worker count: " << worker_count;
      }
    }
    return Status::OK();
  } else {
    // We're not scaling --> We're in a period of stability
    uint64 same_scale_counter;
    TF_RETURN_IF_ERROR(metadata_store.IncrementSameScaleCounter(job_id,
      same_scale_counter));

//    if (same_scale_counter > kInStabilityBeforeScaling) {
//
//    }

  }


  // TODO Check if split provider reached eos, in which case there is no point to scale up.
  worker_count = metrics_history[metrics_history.size()-1]->worker_count(); // guaranteed to work.
  metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
  return Status::OK();
}



} // namespace scaling_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow


