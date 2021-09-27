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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {
namespace easl {

class InterArrivalTimeRepo {
 public:
  // Times to be added in ms
  static void AddInterArrivalTime(double x) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    EnsureMeasurementsInitialized();
    InterArrivalTimeRepo::inter_arrival_times_ms_.pop_front();
    InterArrivalTimeRepo::inter_arrival_times_ms_.push_back(x);
  }
  
  // Average time to be returned in ms
  static double GetAverageInterArrivalTime() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    EnsureMeasurementsInitialized();
    double total = 0.0;
    for (double val : inter_arrival_times_ms_) {
      total += val;
    }
    return total / inter_arrival_times_ms_.size();
  }

 private:
  static void EnsureMeasurementsInitialized() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (inter_arrival_times_ms_.size() == 0) {
      inter_arrival_times_ms_ = std::deque<double>(measurement_count_, 0.0);
    }
  }

  static mutex mu_;
  const static int32 measurement_count_ = 20;
  static std::deque<double> inter_arrival_times_ms_ TF_GUARDED_BY(mu_); 
};

} // easl
} // data
} // tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_EASL_INTER_ARRIVAL_TIME_H_