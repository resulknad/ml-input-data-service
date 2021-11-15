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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_INTER_ARRIVAL_TIME_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_INTER_ARRIVAL_TIME_H_

#include <deque>
#include <limits>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {
namespace easl {

class InterArrivalTimeRepo {
 public:
  // Singleton accessor
  static InterArrivalTimeRepo& GetInstance() {
    static InterArrivalTimeRepo singleton;
    return singleton; 
  }
  
  // Times to be added in ms
  void AddInterArrivalTime(uint64 x, int32 thread_id = 0) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    VLOG(0) << "(AddInterArrivalTime) This should not be printed";
    if (!times_.contains(thread_id)) {
      times_[thread_id] = std::deque<uint64>();
    }
    times_[thread_id].push_back(x);
  }
  
  // Average time to be returned in ms
  double GetAverageInterArrivalTime() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    VLOG(0) << "(GetAverageInterArrivalTime) This should not be printed";
    uint32 min_len = std::numeric_limits<uint32_t>::max();
    for (auto& p : times_) {
      min_len = std::min<uint32>(min_len, p.second.size());
    }

    // Test if no metrics have been collected
    if (min_len == std::numeric_limits<uint32_t>::max() || 
        min_len < min_batches_per_average_) {
      // Return big value to force 1 worker decision
      return last_inter_arrival_time_ms_;
    }

    // Get min inter-arrival across each batch
    last_inter_arrival_time_ms_ = 0.0;
    for (int i = 0; i < min_len; ++i) {
      int32 min_at_batch = -1;
      for (auto& p : times_) {
        if (min_at_batch != -1) {
          min_at_batch = std::min<int32>(min_at_batch, p.second.front());
        } else {
          min_at_batch = p.second.front();
        }
        p.second.pop_front();
      }
      last_inter_arrival_time_ms_ += min_at_batch;
    }

    last_inter_arrival_time_ms_ /= (min_len * times_.size());
    VLOG(0) << "(InterArrivalTimeRepo::GetAverageInterArrivalTime) Time [ms]: " 
            << last_inter_arrival_time_ms_;
    return last_inter_arrival_time_ms_;
  }

 private:

  mutex mu_;
  const uint32 min_batches_per_average_ = 20u;
  double last_inter_arrival_time_ms_ = std::numeric_limits<double>::max();
  absl::flat_hash_map<int32, std::deque<uint64>> times_ TF_GUARDED_BY(mu_);
};

} // easl
} // data
} // tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_INTER_ARRIVAL_TIME_H_