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
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace easl {

// A class encapsulating the metadata store of the tf.data service.
// TODO(damien-aymon)
// - persist this store to disk for fault recovery.

class ClientMetrics {
  public:
    explicit ClientMetrics() : metrics_() {}

    class Metrics {
      public:
        explicit Metrics(double get_next_time, double inter_arrival_time) 
          : get_next_time_(get_next_time),
            inter_arrival_time_(inter_arrival_time) {}

      private:
        double get_next_time_;
        double inter_arrival_time_;
    };

    // The keys are the client id
    absl::flat_hash_map<int64, std::shared_ptr<Metrics>> metrics_; 
};


class WorkerMetrics {
  public:
    WorkerMetrics() : metrics_() {}

    class Metrics {
      public:
        explicit Metrics(int64 bytes_consumed, int64 bytes_produced, 
                         int64 num_elements, int64 computation_time, 
                         double in_node_time, double in_prefix_time) :
                         bytes_consumed_(bytes_consumed),
                         bytes_produced_(bytes_produced),
                         num_elements_(num_elements),
                         computation_time_(computation_time),
                         in_node_time_(in_node_time),
                         in_prefix_time_(in_prefix_time) {}
        
        void set_bytes_consumed(int64 x)   { bytes_consumed_ = x; }
        void set_bytes_produced(int64 x)   { bytes_produced_ = x; }
        void set_num_elements(int64 x)     { num_elements_ = x; }
        void set_computation_time(int64 x) { computation_time_ = x; }
        void set_in_node_time(double x)    { in_node_time_ = x; }
        void set_in_prefix_time(double x)  { in_prefix_time_ = x; }

        int64 get_bytes_consumed()   { return bytes_consumed_; }
        int64 get_bytes_produced()   { return bytes_produced_; }
        int64 get_num_elements()     { return num_elements_; }
        int64 get_computation_time() { return computation_time_; }
        int64 get_in_node_time()     { return in_node_time_; }
        int64 get_in_prefix_time()   { return in_prefix_time_; }

      private:
        int64 bytes_consumed_;
        int64 bytes_produced_;
        int64 num_elements_;
        int64 computation_time_;
        double in_node_time_;
        double in_prefix_time_; 
    };

    // The keys are the worker addresses
    absl::flat_hash_map<std::string, std::shared_ptr<Metrics>> metrics_;
};

class JobMetrics {
  public:
    JobMetrics(int64 job_id, uint64 pipeline_fingerprint) 
      : job_id_(job_id),
        pipeline_fingerprint_(pipeline_fingerprint),
        client_metrics_(), 
        worker_metrics_() {}

    int64 job_id_;
    uint64 pipeline_fingerprint_;
    std::shared_ptr<ClientMetrics> client_metrics_;
    std::shared_ptr<WorkerMetrics> worker_metrics_;
};

class MetadataStore {
 public:
  MetadataStore();
  MetadataStore(const MetadataStore &) = delete;
  MetadataStore &operator=(const MetadataStore &) = delete;

  // Returns the metrics for a job in the `metrics` parameter
  Status GetJobMetrics(int64 job_id, std::shared_ptr<JobMetrics> metrics) const;

  // Update or create the metrics for a client
  // Status UpdateClientMetrics(int64 job_id, int64 client_id, 
  //                            const ClientMetrics::Metrics& metrics);

  // Update or create the metrics for a client
  // Status UpdateWorkerMetrics(int64 job_id, string worker_address, 
  //                            const WorkerMetrics::Metrics& metrics);
 private:
  // Key is job id
  absl::flat_hash_map<int64, std::shared_ptr<JobMetrics>> metadata_;

};

} // namespace easl
} // namespace data
} // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_METADATA_STORE_H_
