#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

ClientMetrics::Metrics::Metrics(double get_next_time, double inter_arrival_time) 
  : get_next_time_(get_next_time),
    inter_arrival_time_(inter_arrival_time) {}

ClientMetrics::Metrics::Metrics(ClientMetrics::Metrics& other) 
  : get_next_time_(other.get_next_time()),
    inter_arrival_time_(other.inter_arrival_time()) {}

void ClientMetrics::Metrics::Update(ClientMetrics::Metrics& other) {
  get_next_time_ = other.get_next_time_;
  inter_arrival_time_ = other.inter_arrival_time_;
}

WorkerMetrics::Metrics::Metrics(WorkerMetrics::Metrics& other) 
  : bytes_consumed_(other.bytes_consumed()),
    bytes_produced_(other.bytes_produced()),
    num_elements_(other.num_elements()),
    computation_time_(other.computation_time()),
    in_node_time_(other.in_node_time()),
    in_prefix_time_(other.in_prefix_time()) {}

WorkerMetrics::Metrics::Metrics(int64 bytes_consumed, int64 bytes_produced, 
  int64 num_elements, int64 computation_time, double in_node_time, 
  double in_prefix_time) 
  : bytes_consumed_(bytes_consumed),
    bytes_produced_(bytes_produced),
    num_elements_(num_elements),
    computation_time_(computation_time),
    in_node_time_(in_node_time),
    in_prefix_time_(in_prefix_time) {}

void WorkerMetrics::Metrics::Update(WorkerMetrics::Metrics& other) {
  bytes_consumed_ = other.bytes_consumed_;
  bytes_produced_ = other.bytes_produced_;
  num_elements_ = other.num_elements_;
  computation_time_ = other.computation_time_;
  in_node_time_ = other.in_node_time_;
  in_prefix_time_ = other.in_prefix_time_; 
}

JobMetrics::JobMetrics(int64 job_id, int64 dataset_id) 
      : job_id_(job_id),
        dataset_id_(dataset_id),
        client_metrics_(), 
        worker_metrics_() {}

MetadataStore::MetadataStore() : metadata_() {}

Status MetadataStore::GetJobMetrics(int64 job_id, 
  std::shared_ptr<JobMetrics> metrics) const {
  auto it = metadata_.find(job_id);
  if (it == metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

Status MetadataStore::GetClientMetrics(int64 job_id, int64 client_id, 
  std::shared_ptr<ClientMetrics::Metrics> metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    auto it = job_metrics->client_metrics_->metrics_.find(client_id);
    if (it == job_metrics->client_metrics_->metrics_.end()) {
      return errors::NotFound("Client with id ", client_id, " within job "
        "with id ", job_id, " does not have metrics");
    }
    metrics = it->second;
  }
  return s;
}

Status MetadataStore::GetWorkerMetrics(int64 job_id, string worker_address, 
    std::shared_ptr<WorkerMetrics::Metrics> metrics) const {
  std::shared_ptr<JobMetrics> job_metrics;
  Status s = GetJobMetrics(job_id, job_metrics);
  if (s.ok()) {
    auto it = job_metrics->worker_metrics_->metrics_.find(worker_address);
    if (it == job_metrics->worker_metrics_->metrics_.end()) {
      return errors::NotFound("Worker with address ", worker_address, " within "
        "job with id ", job_id, " does not have metrics");
    }
    metrics = it->second;
  }
  return s;
}

Status MetadataStore::UpdateClientMetrics(int64 job_id, int64 client_id, 
  ClientMetrics::Metrics& metrics) {
  std::shared_ptr<ClientMetrics::Metrics> current_metrics;
  Status s = GetClientMetrics(job_id, client_id, current_metrics);
  if (s.ok()) {
    current_metrics->Update(metrics);
  } else {
    // This might be the first time this client gets metrics
    std::shared_ptr<JobMetrics> job_metrics;
    s = GetJobMetrics(job_id, job_metrics);
    if (s.ok()) {
      std::shared_ptr<ClientMetrics::Metrics> new_client_metrics = 
        std::make_shared<ClientMetrics::Metrics>(metrics);
      job_metrics->client_metrics_->metrics_.insert({client_id, new_client_metrics});
    }
  }
  return s;
}

Status MetadataStore::UpdateWorkerMetrics(int64 job_id, string worker_address, 
  WorkerMetrics::Metrics& metrics) {
  std::shared_ptr<WorkerMetrics::Metrics> current_metrics;
  Status s = GetWorkerMetrics(job_id, worker_address, current_metrics);
  if (s.ok()) {
    current_metrics->Update(metrics);
  } else {
    // This might be the first time this worker gets metrics
    std::shared_ptr<JobMetrics> job_metrics;
    s = GetJobMetrics(job_id, job_metrics);
    if (s.ok()) {
      std::shared_ptr<WorkerMetrics::Metrics> new_worker_metrics = 
        std::make_shared<WorkerMetrics::Metrics>(metrics);
      job_metrics->worker_metrics_->metrics_.insert({worker_address, new_worker_metrics});
    }
  }
  return s;
}

Status MetadataStore::CreateJob(int64 job_id, int64 dataset_id) {
  auto it = metadata_.find(job_id);
  if (it == metadata_.end()) {
    std::shared_ptr<JobMetrics> job_metrics = std::make_shared<JobMetrics>(
      job_id, dataset_id);
    metadata_.insert({job_id, job_metrics});
  }
  return Status::OK();
}

Status MetadataStore::RemoveJob(int64 job_id) {
  int delete_count = metadata_.erase(job_id);
  return Status::OK();
}

} // namespace easl
} // namespace data
} // namespace tensorflow
