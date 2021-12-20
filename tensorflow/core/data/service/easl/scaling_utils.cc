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

double kMinBatchTimeRelativeImprovement = 0.02; // 2%
uint32 kInStabilityBeforeScaling = 20;
double kMinQueueSizeRelativeGrowth = 1.1; // +10%
double kMinBatchTimeRelativeGrowth = 1.1; // +10%


double kLastBatchCount = 50;
double kEfficiency = 0.95;
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
  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Worker count for last metrics: "
               << metrics_history[metrics_history.size()-1]->worker_count(); // Guaranteed to succeed.

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

    int64 current_target_worker_count;
    TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id, current_target_worker_count));
    if (last_metrics->worker_count() != current_target_worker_count) {
      VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Target metrics count not fulfilled: "
                   << " > target: " << current_target_worker_count
                   << " > actual: " << last_metrics->worker_count();
      worker_count = current_target_worker_count;
      return Status::OK();
    }

    int second_to_last_index = metrics_history.size() - 2;
    while(second_to_last_metrics->worker_count() == last_metrics->worker_count()){
      if (second_to_last_index == 0) {
        VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Should not enter here!"
        << "This might lead to an infinite loop! ";
        worker_count = metrics_history.back()->worker_count();
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        return Status::OK();
      }
      second_to_last_metrics = metrics_history[--second_to_last_index];
    }

    // FIXME: Using a constant here for the last batch count; make sure
    //        it actually matches the metric in the client
    double previous_throughput = kLastBatchCount /
      second_to_last_metrics->last_x_batch_time_ms();
    double current_throughput = kLastBatchCount /
      last_metrics->last_x_batch_time_ms();

    if (second_to_last_metrics->worker_count() < last_metrics->worker_count()) {
      // We are scaling up
      double projected_throughput = (previous_throughput *
        last_metrics->worker_count()) / second_to_last_metrics->worker_count();
      double efficiency = current_throughput / projected_throughput;

      if (efficiency >= kEfficiency){
        worker_count = last_metrics->worker_count() + 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                     << "Improvement large enough:\n"
                     << " > efficiency: " << efficiency << "\n"
                     << " > next worker count: " << worker_count;
      } else {
        worker_count = second_to_last_metrics->worker_count();
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingUp) "
                << "Improvement NOT large enough:\n"
                << " > efficiency: " << efficiency << "\n"
                << " > next worker count: " << worker_count;
      }
    } else {
      // We are scaling down
      double projected_throughput = (current_throughput *
        second_to_last_metrics->worker_count()) / last_metrics->worker_count();
      double efficiency = previous_throughput / projected_throughput;

      if (efficiency < kEfficiency) {
        worker_count = last_metrics->worker_count() - 1;
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss ok:\n"
                << " > efficiency: " << efficiency << "\n"
                << " > next worker count: " << worker_count;
      } else {
        worker_count = second_to_last_metrics->worker_count();
        metadata_store.UnsetJobIsScaling(job_id);
        metadata_store.ResetSameScaleCounter(job_id);
        VLOG(0) << "(EASL::DynamicWorkerCountUpdate::ScalingDown) "
                << "Improvement loss NOT ok:\n"
                << " > efficiency: " << efficiency << "\n"
                << " > next worker count: " << worker_count;
      }
    }
    metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
    return Status::OK();
  } else {
    // We're not scaling --> We're in a period of stability
    uint64 same_scale_counter;
    TF_RETURN_IF_ERROR(metadata_store.IncrementSameScaleCounter(job_id,
      same_scale_counter));

    if (same_scale_counter == 1){
      // Set the converged metrics
      int64 target_worker_count;
      TF_RETURN_IF_ERROR(metadata_store.GetJobTargetWorkerCount(job_id, target_worker_count));
      int converged_index = metrics_history.size() - 1;
      while(metrics_history[converged_index]->worker_count() != target_worker_count) {
        if (converged_index == 0) {
          VLOG(0)
          << "EASL (DynamicWorkerCountUpdate) - Did not find metrics for target_worker_count, using oldest instead";
          break;
        }
        --converged_index;
      }
      model_metrics->converged_metrics_ = metrics_history[converged_index];
    }

    if (same_scale_counter > kInStabilityBeforeScaling) {
      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
          << "Checking for potential rescaling after stable period.";
      metadata_store.ResetSameScaleCounter(job_id);
      std::shared_ptr<ModelMetrics::Metrics> converged_metrics = model_metrics->converged_metrics_;
      std::shared_ptr<ModelMetrics::Metrics> last_metrics = metrics_history[metrics_history.size() - 1];

      double queue_size_converged = converged_metrics->result_queue_size();
      double queue_size_last_metrics = last_metrics->result_queue_size();
      double relative_queue_size = queue_size_last_metrics / queue_size_converged;

      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
          << "relative_queue_size: " << relative_queue_size << "\n"
          << "queue_size_converged: " << queue_size_converged << "\n"
          << "queue_size_last_metrics: " << queue_size_last_metrics << "\n";

      double converged_batch_time = converged_metrics->last_x_batch_time_ms();
      double l_batch_time = last_metrics->last_x_batch_time_ms();
      double relative_batch_time = l_batch_time / converged_batch_time;

      VLOG(0) << "(EASL::DynamicWorkerCountUpdate::StablePeriod) - "
              << "relative_batch_time: " << relative_batch_time << "\n"
              << "converged_batch_time: " << converged_batch_time << "\n"
              << "l_batch_time: " << l_batch_time << "\n";

      if (relative_queue_size > kMinQueueSizeRelativeGrowth){
        VLOG(0) << "Triggering downscale";
        worker_count = last_metrics->worker_count() - 1;
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        metadata_store.SetJobIsScaling(job_id);
        return Status::OK();
      }

      if (relative_batch_time > kMinBatchTimeRelativeGrowth){
        VLOG(0) << "Triggering upscale";
        worker_count = last_metrics->worker_count() + 1;
        metadata_store.SetJobTargetWorkerCount(job_id, worker_count);
        metadata_store.SetJobIsScaling(job_id);
        return Status::OK();
      }

      VLOG(0) << "No rescale triggered";

      worker_count = last_metrics->worker_count();
      return Status::OK();
    }
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


