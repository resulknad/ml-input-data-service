//
// Created by Muyu Li on 16.11.21.
//

#include "local_decision_utils.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_decision {

Status DecideIfLocal(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        bool& using_local_workers) {
    using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
    using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

    // Check if we have any metrics for this dataset
    std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
    Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(
            dataset_key, job_metrics);

    // We do not yet have the metrics for this dataset --> use 1 worker
    if(errors::IsNotFound(s)) {
        VLOG(0) << "DSL (DecideIfLocal) No metrics found for dataset, will use normal mode";
        using_local_workers = false;
        return Status::OK();
    } else if (!s.ok()) {
        VLOG(0) << "DSL (DecideIfLocal) Another error has been thrown: " << s;
        return s;
    }

    // Pipeline stats: last TF node metrics
    std::shared_ptr<NodeMetrics> last_tf_node_metrics;

    s = metadata_store.GetLastNodeMetricsByDatasetKey(dataset_key, last_tf_node_metrics);
    if (!s.ok()) {
        VLOG(0) << "DSL (DecideIfLocal) Failed to get the last TF node metrics";
        return s;
    }

    int64_t total_bytes_produced = 0, total_num_elements = 0;
    for (std::pair<string, std::shared_ptr<NodeMetrics::Metrics>> e :
            last_tf_node_metrics->metrics_) {
        std::shared_ptr<NodeMetrics::Metrics> node_metrics = e.second;
        total_bytes_produced += node_metrics->bytes_produced();
        total_num_elements += node_metrics->num_elements();
    }

    double avg_bytes_per_element = (double)total_bytes_produced / total_num_elements;
    VLOG(0) << "DSL (DecideIfLocal) Total bytes produced: " << total_bytes_produced << "\n"
            << "Total num elements: " << total_num_elements << "\n"
            << "Avg bytes produced per element: " << avg_bytes_per_element << "\n"
            << "Decision Threshold: " << dispatcher_config.avg_bytes_per_element_local_thres() << "\n";

    if (avg_bytes_per_element > dispatcher_config.avg_bytes_per_element_local_thres()) {
        using_local_workers = true;
        VLOG(0) << "DSL (DecideIfLocal) Using local workers! (because avg. bytes per element > threshold) \n";
    }
    else {
        using_local_workers = false;
        VLOG(0) << "DSL (DecideIfLocal) NOT using local workers! (because avg. bytes per element < threshold) \n";
    }

    return Status::OK();
}

Status DecideTargetWorkers(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        int64 num_worker_remote_avail,
        int64 num_worker_local_avail,
        int64& num_worker_remote_target,
        int64& num_worker_local_target) {
  bool using_local_workers;
  TF_RETURN_IF_ERROR(service::easl::local_decision::DecideIfLocal(
      dispatcher_config, metadata_store, dataset_key, using_local_workers
      ));

  if(using_local_workers) {
      num_worker_remote_target = num_worker_remote_avail / 2;
      num_worker_local_target = num_worker_local_avail / 2;
  } else {
      num_worker_remote_target = num_worker_remote_avail / 2;
      num_worker_local_target = 0;
  }

  VLOG(0) << "DSL (DecideTargetWorkers) Available remote: " << num_worker_remote_avail << "\n"
          << "Available local: " << num_worker_local_avail << "\n"
          << "Decided remote: " << num_worker_remote_target << "\n"
          << "Decided local: " << num_worker_local_target << "\n";

  return Status::OK();
}

} // namespace local_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow