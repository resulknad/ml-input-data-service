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
    {
      mutex_lock l(mu_);
      thread_ids_.insert(thread_id);
    }

    // Prevent overflow
    if ((max_val_ - x) < accumulator_ms_.load()) {
      accumulator_ms_.store(0u);
      measurement_count_.store(0u);
    }

    accumulator_ms_ += x;
    ++measurement_count_;
  }
  
  // Average time to be returned in ms
  double GetAverageInterArrivalTime() TF_LOCKS_EXCLUDED(mu_) {
    uint32 thread_count = 1;
    {
      mutex_lock l(mu_);
      uint32 thread_count = std::max<uint32>(thread_ids_.size(), 1);
    }
    double average = (double)(accumulator_ms_.load()) / 
      (measurement_count_ * thread_count);
    VLOG(0) << "(InterArrivalTimeRepo::GetAverageInterArrivalTime) Time [ms]: " 
            << average;
    return average;
  }

 private:
  InterArrivalTimeRepo() : accumulator_ms_(0u), measurement_count_(0u) {}

  mutex mu_;
  std::set<int32> thread_ids_ TF_GUARDED_BY(mu_);
  std::atomic_uint64_t accumulator_ms_;
  std::atomic_uint64_t measurement_count_;
  const uint64 max_val_ = std::numeric_limits<uint64>::max();
};

} // easl
} // data
} // tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_INTER_ARRIVAL_TIME_H_