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

  std::shared_ptr<ModelMetrics::Metrics> last_model_metrics =
      job_metrics->model_metrics_->metrics_history_.back();
  worker_count = last_model_metrics->worker_count();

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
    const int64 current_worker_count,
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
    worker_count = current_worker_count;
    if ( counter > 20){
      if (current_worker_count < 2){
        worker_count = 2;
      } else {
        worker_count = 1;
      }
      counter = 0;
    }
    VLOG(0) << "EASL - Dynamic scaling, counter " << counter
            << ", current_worker_count " << current_worker_count
            << ", worker_count " << worker_count;
    return Status::OK();
  }

  VLOG(0) << "EASL (DynamicWorkerCountUpdate) - Entering. Current worker count "
  << current_worker_count;

  bool is_scaling;
  TF_RETURN_IF_ERROR(metadata_store.IsJobScaling(job_id, is_scaling));

  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetModelMetrics(job_id, model_metrics));

  ModelMetrics::MetricsHistory metrics_history = model_metrics->metrics_history_;

  if(is_scaling) {
    VLOG(0) << "EASL (DynamicWorkerCountUpdate) - is_scaline is true";
    if (metrics_history.size() == 1){ // Cannot be smaller than 1
      VLOG(0) << "EASL (DynamicWorkerCountUpdate) - no metrics_history -> increasing worker count";
      worker_count = metrics_history.back()->worker_count() + 1;
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
        return Status::OK();
      }
      second_to_last_metrics = metrics_history[--second_to_last_index];
    }

    double stl_batch_time = second_to_last_metrics->last_x_batch_time_ms();
    double l_batch_time = last_metrics->last_x_batch_time_ms();
    double relative_improvement = 1.0 - l_batch_time / stl_batch_time;

    if (second_to_last_metrics->worker_count() < last_metrics->worker_count()){
      // We are scaling up
      if (relative_improvement > kMinBatchTimeRelativeImprovement){
        worker_count = last_metrics->worker_count() + 1;
      } else {
        worker_count = last_metrics->worker_count();
        metadata_store.UnsetJobIsScaling(job_id);
      }
    }
  }



  // TODO Check if split provider reached eos, in which case there is no point to scale up.
  worker_count = current_worker_count;
  return Status::OK();

  /**

  // Check if we have any metrics for this dataset
  std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
  Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(
      dataset_key, job_metrics);

  // We do not yet have the metrics for this dataset --> use 1 worker
  if(errors::IsNotFound(s)) {
    VLOG(0) << "(DynamicWorkerCountUpdate) No metrics found for dataset "
            << dataset_key << ". Will use 1 worker in " << job_type << " mode.";
    worker_count = 1;
    return Status::OK();
  } else if (!s.ok()) {
    VLOG(0) << "(DynamicWorkerCountUpdate) Another error has been thrown: " << s;
    return s;
  }

  // Pipeline stats: last TF node metrics
  std::shared_ptr<NodeMetrics> last_tf_node_metrics;
  s = metadata_store.GetLastTFNodeMetricsByDatasetKey(
      dataset_key, last_tf_node_metrics);
  if (!s.ok()) {
    VLOG(0) << "(DetermineElasticity) Failed to get the last TF node metrics";
    return s;
  }
  size_t num_workers = (last_tf_node_metrics->metrics_).size();
  DCHECK(num_workers > 0);

  // Model stats
  double client_throughput = 0.0;
  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(
      metadata_store.GetModelMetricsByDatasetKey(
          dataset_key, model_metrics));

  // Get the average client throughput
  for(std::pair<int64, std::shared_ptr<ModelMetrics::Metrics>> e :
      model_metrics->metrics_) {
    std::shared_ptr<ModelMetrics::Metrics> client_metrics = e.second;
    client_throughput += 1000.0 / client_metrics->inter_arrival_time_ms();
  }
  // Multiply the average throughput by the nr of clients to get the real throughput
  VLOG(0) << "(DetermineElasticity) Total client throughput demand "
          << client_throughput;

  if (job_type == "COMPUTE" || job_type == "PUT") {
    VLOG(0) << "(DetermineElasticity) In COMPUTE or PUT branch with case "
            << job_type;
    double avg_worker_throughput = 0.0;

    // Get the average throughput for a worker
    for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e :
        last_tf_node_metrics->metrics_) {
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
      avg_worker_throughput += 1000.0 / worker_metrics->active_time_ms();
    }
    avg_worker_throughput /= num_workers;
    VLOG(0) << "(DetermineElasticity) Average worker throughput "
            << avg_worker_throughput;

    // Infer the number of workers required to sustain the model
    worker_count = std::max<int64>(ceil(client_throughput / avg_worker_throughput -
        worker_count_alpha_), 1);
  } else {
    VLOG(0) << "(DetermineElasticity) In GET branch.";

    // Get last user node metrics
    std::shared_ptr<NodeMetrics> last_node_metrics;
    TF_RETURN_IF_ERROR(
        metadata_store.GetLastNodeMetricsByDatasetKey(
            dataset_key, last_node_metrics));
    DCHECK(num_workers == (last_node_metrics->metrics_).size());

    // Get the time per row and the TF nodes overhead
    uint64 row_size = 0;
    double tf_nodes_overhead_ms = 0.0;

    for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e :
        last_node_metrics->metrics_) {
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics_tf_node =
          last_tf_node_metrics->metrics_[e.first];

      row_size += worker_metrics->bytes_produced() /
          worker_metrics->num_elements();
      tf_nodes_overhead_ms += worker_metrics_tf_node->active_time_ms()
          - worker_metrics->active_time_ms();
    }

    row_size /= num_workers;
    tf_nodes_overhead_ms /= num_workers;
    double cache_read_time_per_row_ms = data::cache_model::GetTimePerRow(
        row_size);

    VLOG(0) << "(DetermineElasticity) Average row size " << row_size;
    VLOG(0) << "(DetermineElasticity) Average read time per item "
            << cache_read_time_per_row_ms;
    VLOG(0) << "(DetermineElasticity) Average TF overhead "
            << tf_nodes_overhead_ms;

    // Infer the worker count for the cache GET use case
    worker_count = std::max<int64>(ceil(client_throughput *
        (cache_read_time_per_row_ms + tf_nodes_overhead_ms)
                                            / 1000.0 - worker_count_alpha_), 1);
  }
  VLOG(0) << "(DetermineElasticity) Inferred workers " << worker_count;

  return Status::OK();
   */
}



} // namespace scaling_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow


