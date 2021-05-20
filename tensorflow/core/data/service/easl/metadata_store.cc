#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {


// Metrics from the Client
ModelMetrics::Metrics::Metrics(double get_next_time, double inter_arrival_time) 
  : get_next_time_(get_next_time),
    inter_arrival_time_(inter_arrival_time) {}

ModelMetrics::Metrics::Metrics(ModelMetrics::Metrics& other) 
  : get_next_time_(other.get_next_time()),
    inter_arrival_time_(other.inter_arrival_time()) {}

void ModelMetrics::Metrics::Update(ModelMetrics::Metrics& other) {
  get_next_time_ = other.get_next_time_;
  inter_arrival_time_ = other.inter_arrival_time_;
}

Status ModelMetrics::UpdateClientMetrics(int64 client_id, 
  ModelMetrics::Metrics& metrics) {
  auto it = metrics_.find(client_id);
  if (it != metrics_.end()) {
    it->second->Update(metrics);
  } else {
    auto entry = std::make_shared<Metrics>(metrics);
    metrics_.insert({client_id, entry});
  }
  return Status::OK();
}

Status ModelMetrics::GetClientMetrics(int64 client_id, Metrics* metrics) {
  auto it = metrics_.find(client_id);
  if (it != metrics_.end()) {
    metrics = it->second.get();
    return Status::OK();
  }
  return errors::NotFound("No metrics under the client with id ", client_id);
}

// Metrics from the Worker Nodes
NodeMetrics::Metrics::Metrics(NodeMetrics::Metrics& other) 
  : bytes_consumed_(other.bytes_consumed()),
    bytes_produced_(other.bytes_produced()),
    num_elements_(other.num_elements()),
    computation_time_(other.computation_time()),
    in_node_time_(other.in_node_time()),
    in_prefix_time_(other.in_prefix_time()) {}

NodeMetrics::Metrics::Metrics(int64 bytes_consumed, int64 bytes_produced, 
  int64 num_elements, int64 computation_time, double in_node_time, 
  double in_prefix_time) 
  : bytes_consumed_(bytes_consumed),
    bytes_produced_(bytes_produced),
    num_elements_(num_elements),
    computation_time_(computation_time),
    in_node_time_(in_node_time),
    in_prefix_time_(in_prefix_time) {}

void NodeMetrics::Metrics::Update(NodeMetrics::Metrics& other) {
  bytes_consumed_ = other.bytes_consumed_;
  bytes_produced_ = other.bytes_produced_;
  num_elements_ = other.num_elements_;
  computation_time_ = other.computation_time_;
  in_node_time_ = other.in_node_time_;
  in_prefix_time_ = other.in_prefix_time_; 
}

Status NodeMetrics::UpdateWorkerMetrics(string worker_address, 
  NodeMetrics::Metrics& metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    it->second->Update(metrics);
  } else {
    auto entry = std::make_shared<NodeMetrics::Metrics>(metrics);
    metrics_.insert({worker_address, entry});
  }
  return Status::OK();
}

Status NodeMetrics::GetWorkerMetrics(string worker_address, Metrics* metrics) {
  auto it = metrics_.find(worker_address);
  if (it != metrics_.end()) {
    metrics = it->second.get();
    return Status::OK();
  }
  return errors::NotFound("No metrics under the worker with address ", 
    worker_address);
}

// Input pipeline metrics
Status InputPipelineMetrics::GetNodeMetrics(string long_name, 
  NodeMetrics* metrics) {
  auto it = metrics_.find(long_name);
  if (it != metrics_.end()) {
    metrics = it->second.get();
    return Status::OK();
  }
  return errors::NotFound("No metrics for node ", long_name); 
}

Status InputPipelineMetrics::GetWorkerMetrics(string worker_address, 
  absl::flat_hash_map<string, NodeMetrics::Metrics*>& metrics) {
  for (auto& entry : metrics_) {
    NodeMetrics::Metrics* node_metrics;
    Status s = entry.second->GetWorkerMetrics(worker_address, node_metrics);
    if (s.ok()) {
      metrics.insert({entry.first, node_metrics});
    }
  }
  return Status::OK();
}

Status InputPipelineMetrics::UpdateNodeMetrics(string long_name, 
  string worker_address, NodeMetrics::Metrics& metrics) {
  VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Trying to find " <<
    long_name;
  auto it = metrics_.find(long_name); 
  if (it == metrics_.end()) {
    VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Not found, now adding";
    auto node_metrics = std::make_shared<NodeMetrics>();
    node_metrics->UpdateWorkerMetrics(worker_address, metrics);
    VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Done 1.1";
    metrics_.insert({long_name, node_metrics});
    VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Done 1.2";
  } else {
    VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Found, now updating";
    it->second->UpdateWorkerMetrics(worker_address, metrics);
    VLOG(2) << "(InputPipelineMetrics::UpdateNodeMetrics) Done 2.1";
  }
  return Status::OK();
}

// Job metrics
JobMetrics::JobMetrics(int64 job_id, int64 dataset_id) 
      : job_id_(job_id),
        dataset_id_(dataset_id),
        model_metrics_(std::make_shared<ModelMetrics>()), 
        input_pipeline_metrics_(std::make_shared<InputPipelineMetrics>()) {}

// Metadata store 
Status MetadataStore::CreateJob(int64 job_id, int64 dataset_id) {
  auto it = metadata_.find(job_id);
  if (it == metadata_.end()) {
    auto job_metrics = std::make_shared<JobMetrics>(job_id, dataset_id);
    metadata_.insert({job_id, job_metrics});
    VLOG(2) << "(MetadataStore::CreateJob) Created job with id " << job_id 
            << " and dataset id " << dataset_id;
  }
  return Status::OK();
}

Status MetadataStore::RemoveJob(int64 job_id) {
  metadata_.erase(job_id);
  VLOG(2) << "(MetadataStore::RemoveJob) Removed job with id " << job_id;
  return Status::OK();
}

Status MetadataStore::GetJobMetrics(int64 job_id, JobMetrics* metrics) const {
  auto it = metadata_.find(job_id);
  if (it == metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  metrics = it->second.get();
  return Status::OK();
}

Status MetadataStore::GetModelMetrics(int64 job_id, 
  ModelMetrics* metrics) const {
  JobMetrics* job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->model_metrics_.get();
  }
  return s;
}

Status MetadataStore::GetInputPipelineMetrics(int64 job_id, 
  InputPipelineMetrics* metrics) const {
  JobMetrics* job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    metrics = job_metrics->input_pipeline_metrics_.get();
  }
  return s;
}

Status MetadataStore::UpdateModelMetrics(int64 job_id, int64 client_id, 
  ModelMetrics::Metrics& metrics) {
  ModelMetrics* model_metrics;
  Status s = GetModelMetrics(job_id, model_metrics);
  if (s.ok()) {
    model_metrics->UpdateClientMetrics(client_id, metrics);
  } 
  // if s != ok --> no such job exists
  return s;
}

Status MetadataStore::UpdateInputPipelineMetrics(int64 job_id, 
  string node_long_name, string worker_address, NodeMetrics::Metrics& metrics) {
  InputPipelineMetrics* pipeline_metrics;
  Status s = GetInputPipelineMetrics(job_id, pipeline_metrics);
  VLOG(2) << "(MetadataStore::UpdateInputPipelineMetrics) Updating " << job_id;
  if (s.ok()) {
    VLOG(2) << "(MetadataStore::UpdateInputPipelineMetrics) Found the entry";
    pipeline_metrics->UpdateNodeMetrics(node_long_name, worker_address, 
      metrics);
    VLOG(2) << "(MetadataStore::UpdateInputPipelineMetrics) Updated the entry successfully";
  } 
  // if s != ok --> no such job exists
  return s;
}

} // namespace easl
} // namespace data
} // namespace tensorflow
