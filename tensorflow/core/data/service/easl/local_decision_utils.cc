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
        bool& if_local) {
    using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
    using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

    // Check if we have any metrics for this dataset
    std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
    Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(
            dataset_key, job_metrics);

    // We do not yet have the metrics for this dataset --> use 1 worker
    if(errors::IsNotFound(s)) {
        VLOG(0) << "(DecideIfLocal) No metrics found for dataset, will use normal mode";
        if_local = false;
        return Status::OK();
    } else if (!s.ok()) {
        VLOG(0) << "(DecideIfLocal) Another error has been thrown: " << s;
        return s;
    }

    // Pipeline stats: last TF node metrics
    std::shared_ptr<NodeMetrics> last_tf_node_metrics;

//    s = metadata_store.GetLastTFNodeMetricsByDatasetKey(
//            dataset_key, last_tf_node_metrics);
    s = metadata_store.GetLastNodeMetricsByDatasetKey(
            dataset_key, last_tf_node_metrics
            );
    if (!s.ok()) {
        VLOG(0) << "(DecideIfLocal) Failed to get the last TF node metrics";
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
    VLOG(0) << "(DecideIfLocal) Total bytes produced: " << total_bytes_produced << "\n"
            << "Total num elements: " << total_num_elements << "\n"
            << "Avg bytes produced per element: " << avg_bytes_per_element << "\n"
            << "Decision Threshold: " << dispatcher_config.avg_bytes_per_element_local_thres() << "\n";

    if (avg_bytes_per_element > dispatcher_config.avg_bytes_per_element_local_thres()) {
        if_local = true;
        VLOG(0) << "(DecideIfLocal-decision) using local workers\n";
    }
    else {
        if_local = false;
        VLOG(0) << "(DecideIfLocal-decision) using default worker set ";
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
  num_worker_remote_target = num_worker_remote_avail / 2;
  num_worker_local_target = num_worker_local_avail / 2;
  VLOG(1) << "DSL (DecideTargetWorkers) " << num_worker_remote_avail << ' ' << num_worker_local_avail
  << ' ' << num_worker_remote_target << ' ' << num_worker_local_target;
  return Status::OK();
}

} // namespace local_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow