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

#include "tensorflow/core/data/service/split_provider.h"

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {
namespace {
  constexpr char kRepetition[] = "repetition";
  constexpr char kSkipAll[] = "skipall";
  constexpr char kTargetIndex[] = "index";
}
Status DataServiceSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
  mutex_lock l(mu_);
  if (skip_all_) {
    *end_of_splits = true;
    VLOG(0) << "skipping because skip_all_" << skip_all_;
    return Status::OK();
  }
  if (!dispatcher_) {
    dispatcher_ =
        absl::make_unique<DataServiceDispatcherClient>(address_, protocol_);
  }

  tensorflow::Status status;

  // here we skip until target index
  while (index_ <= target_index_) {
    VLOG(0) << "index:" << index_ <<", target_index:" << target_index_;
    status = grpc_util::Retry(
        [this, split, end_of_splits] {
          return dispatcher_->GetSplit(job_id_, task_id_, repetition_,
                                       split_provider_index_, *split,
                                       *end_of_splits);
        },
        "get next split",
        /*deadline_micros=*/Env::Default()->NowMicros() +
            (timeout_ms_ * EnvTime::kMillisToMicros));
    TF_RETURN_IF_ERROR(status);
    index_++;
  }

  target_index_++;
  return status;
}

Status DataServiceSplitProvider::Reset() {
  mutex_lock l(mu_);
  repetition_++;
  index_ = 0;
  target_index_ = 0;
  return Status::OK();
}

Status DataServiceSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  VLOG(0) << "data service split saving";
  auto s1 = (writer->WriteScalar(full_name(kTargetIndex), target_index_));
  VLOG(0) << "post split saving";
  VLOG(0) << "data service split saving, s1:" << s1;
  auto s2 = writer->WriteScalar(full_name(kRepetition), repetition_);
  VLOG(0) << "data service split saving, s1:" << s1 << ", s2:" << s2 << " name: " << full_name(kRepetition) << " val:" << repetition_;
  if (skip_all_) {
    VLOG(0) << "data service split skipall";
    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kSkipAll), skip_all_));
    VLOG(0) << "data service split saving, s1:" << s1 << ", s2:" << s2;
  }
  return s2;
}

Status DataServiceSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  if (initialized_) {
    VLOG(0) << "Refusing to restore twice: " << full_name("") << ". first restore is what matters...";
    return Status::OK();
  }
  initialized_ = true;
  VLOG(0) << "Checking key " << full_name(kRepetition) << " local state: " << target_index_ << ", " << repetition_;
  if (!reader->Contains(full_name(kRepetition)) || reader->Contains(full_name(kSkipAll))) {
    // must have been destroyed when checkpointing...
    // thus we are out of elements -> skip all
    VLOG(0) << "setting skip_all_. " << reader->Contains(full_name(kSkipAll))
      << " because key " << full_name(kRepetition) << "  does not exist;";
    skip_all_ = true; 
    return Status::OK();
  }
  VLOG(0) << "data service split writing";
  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kTargetIndex), &target_index_));
  VLOG(0) << "data service split writing " << target_index_;
  return reader->ReadScalar(full_name(kRepetition), &repetition_);
}

}  // namespace data
}  // namespace tensorflow
