/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace easl {

class ModelMetrics {
  public:
    class Metrics {
      public:
        Metrics(Metrics& other);
        Metrics(double get_next_time, double inter_arrival_time);

        void Update(Metrics& other);

        void set_get_next_time(double x) { get_next_time_ = x; }
        void set_inter_arrival_time(double x) { inter_arrival_time_ = x; }

        double get_next_time() { return get_next_time_; }
        double inter_arrival_time() { return inter_arrival_time_; }

      private:
        double get_next_time_; 
        double inter_arrival_time_;
    };

    using MetricsCollection = 
      absl::flat_hash_map<int64, std::shared_ptr<ModelMetrics::Metrics>>;

    ModelMetrics() : metrics_() {}

    // Update the values for a client
    Status UpdateClientMetrics(int64 client_id, Metrics& metrics);
    Status GetClientMetrics(int64 client_id, std::shared_ptr<Metrics> metrics);

    // The keys are the client id
    MetricsCollection metrics_;
};

class NodeMetrics {
  public: 
    class Metrics {
      public:
        explicit Metrics(Metrics& other);
        explicit Metrics(int64 bytes_consumed, int64 bytes_produced, 
                          int64 num_elements, int64 computation_time, 
                          double in_node_time, double in_prefix_time);
        
        void Update(Metrics& other);
        
        void set_bytes_consumed(int64 x)   { bytes_consumed_ = x; }
        void set_bytes_produced(int64 x)   { bytes_produced_ = x; }
        void set_num_elements(int64 x)     { num_elements_ = x; }
        void set_computation_time(int64 x) { computation_time_ = x; }
        void set_in_node_time(double x)    { in_node_time_ = x; }
        void set_in_prefix_time(double x)  { in_prefix_time_ = x; }

        int64 bytes_consumed()   { return bytes_consumed_; }
        int64 bytes_produced()   { return bytes_produced_; }
        int64 num_elements()     { return num_elements_; }
        int64 computation_time() { return computation_time_; }
        double in_node_time()     { return in_node_time_; }
        double in_prefix_time()   { return in_prefix_time_; }

      private:
        int64 bytes_consumed_;
        int64 bytes_produced_;
        int64 num_elements_;
        int64 computation_time_;
        double in_node_time_;
        double in_prefix_time_; 
    };

    using MetricsCollection =
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics::Metrics>>;

    NodeMetrics() : metrics_() {}

    // Get or update the metrics of a worker
    Status UpdateWorkerMetrics(string worker_address, Metrics& metrics);
    Status GetWorkerMetrics(string worker_address, 
      std::shared_ptr<Metrics> metrics);
  
    // The key here is the worker address
    MetricsCollection metrics_;
};


class InputPipelineMetrics {
  public:
    using MetricsCollection = 
      absl::flat_hash_map<string, std::shared_ptr<NodeMetrics>>;

    InputPipelineMetrics() : metrics_() {}

    // Get the metrics for a single node
    Status GetNodeMetrics(string long_name, 
      std::shared_ptr<NodeMetrics> metrics);

    // Get the metrics from the same worker for each node in the graph 
    Status GetWorkerMetrics(string worker_address, 
      NodeMetrics::MetricsCollection& metrics);

    // Methods for setting data
    Status UpdateNodeMetrics(string long_name, string worker_address, 
      NodeMetrics::Metrics& metrics);

    // The keys are the long name of the node
    MetricsCollection metrics_;
};

class JobMetrics {
  public:
    JobMetrics(int64 job_id, int64 dataset_id);

    int64 job_id_;
    int64 dataset_id_;
    std::shared_ptr<ModelMetrics> model_metrics_;
    std::shared_ptr<InputPipelineMetrics> input_pipeline_metrics_;
};

class MetadataStore {
 public:
  MetadataStore();
  MetadataStore(const MetadataStore &) = delete;
  MetadataStore &operator=(const MetadataStore &) = delete;

  // Create a job entry
  Status CreateJob(int64 job_id, int64 dataset_id);

  // Remove job
  Status RemoveJob(int64 job_id);

  // Get metrics
  Status GetJobMetrics(int64 job_id, std::shared_ptr<JobMetrics> metrics) const;

  Status GetModelMetrics(int64 job_id, 
    std::shared_ptr<ModelMetrics> metrics) const;

  Status GetInputPipelineMetrics(int64 job_id, 
    std::shared_ptr<InputPipelineMetrics> metrics) const;

  // Update or create the metrics for a client
  Status UpdateModelMetrics(int64 job_id, int64 client_id, 
    ModelMetrics::Metrics& metrics);

  // Update or create the metrics for a client
  Status UpdateInputPipelineMetrics(int64 job_id, string node_long_name, 
    string worker_address, NodeMetrics::Metrics& metrics);

 private:
  // Key is job id
  absl::flat_hash_map<int64, std::shared_ptr<JobMetrics>> metadata_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
