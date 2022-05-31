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
#include "tensorflow/core/data/service/task_runner.h"

#include <memory>
#include <vector>

#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/thread_safe_buffer.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {
// Time to wait before skipping a round if data still isn't available.
const int64_t kWaitBeforeSkipUs = 100 * 1000;  // 100ms.

}  // namespace

StandaloneTaskIterator::StandaloneTaskIterator(
    std::unique_ptr<standalone::Dataset> dataset,
    std::unique_ptr<standalone::Iterator> iterator)
    : dataset_(std::move(dataset)), iterator_(std::move(iterator)) {}

Status StandaloneTaskIterator::GetNext(std::vector<Tensor>& element,
                                       bool& end_of_sequence) {
  // SAVING BUSINESS
  // VLOG(0) << "DBK: entering GetNext";
  // VariantTensorDataWriter writer;
  // std::unique_ptr<SerializationContext> serialization_ctx;
  // serialization_ctx =
  // absl::make_unique<SerializationContext>(SerializationContext::Params{});
  // VLOG(0) << "DBK: calling iterator_->Save";
  // TF_RETURN_IF_ERROR(iterator_->Save(serialization_ctx.get(), &writer));

  // VLOG(0) << "DBK: calling get data on writer";
  // std::vector<const VariantTensorData*> data;
  // writer.GetData(&data);
  // VariantTensorDataReader reader(data);

  // this should not be on the iterator
  // restore should be exposed on the standalone dataset...

  // lets leak some memory
  //  iterator_.release();
  /*  VLOG(0) << "DBK: post reset";
    VLOG(0) << "DBK: making split providers";
    std::vector<std::unique_ptr<SplitProvider>> providers;
    TF_RETURN_IF_ERROR(dataset_->MakeSplitProviders(&providers));

    VLOG(0) << "DBK: make iterator from checkpoint";

    TF_RETURN_IF_ERROR(dataset_->MakeIteratorFromCheckpoint(std::move(providers),
    &reader, &iterator_));*/
  //  TF_RETURN_IF_ERROR(iterator_->Restore(&reader, dataset_.get()));

  // VLOG(0) << "DBK: returning get next";
  auto status = iterator_->GetNext(&element, &end_of_sequence);
  // VLOG(0) << "DBK: done with iterator_->getnext";
  return status;
}

int64_t StandaloneTaskIterator::Cardinality() const {
  return dataset_->Get()->Cardinality();
}

Status StandaloneTaskIterator::Save(SerializationContext* ctx,
                                    IteratorStateWriter* writer) {
  return iterator_->Save(ctx, writer);
}

model::Model::ModelMetrics StandaloneTaskIterator::GetMetrics() {
  return iterator_->GetMetrics();
}

Status TaskRunner::Create(const experimental::WorkerConfig& worker_config,
                          const TaskDef& task_def,
                          std::unique_ptr<TaskIterator> iterator,
                          std::unique_ptr<TaskRunner>& out) {
  if (task_def.optional_num_consumers_case() == TaskDef::kNumConsumers) {
    int64_t cardinality = iterator->Cardinality();
    if (cardinality != kInfiniteCardinality &&
        cardinality != kUnknownCardinality) {
      return errors::FailedPrecondition(
          "Round robin reads require that the input dataset has infinite "
          "cardinality, but the dataset has cardinality ",
          cardinality,
          ". Consider adding a `.repeat()` transformation to the dataset.");
    }
    out = absl::make_unique<RoundRobinTaskRunner>(std::move(iterator),
                                                  task_def.num_consumers(),
                                                  task_def.worker_address());
  } else {
    out =
        absl::make_unique<FirstComeFirstServedTaskRunner>(std::move(iterator));
  }
  return Status::OK();
}

Status TaskRunner::CreateFromCheckpoint(
    const experimental::WorkerConfig& worker_config, const TaskDef& task_def,
    std::unique_ptr<TaskIterator> iterator, std::unique_ptr<TaskRunner>& out,
    VariantTensorDataReader* reader) {
  TF_RETURN_IF_ERROR(Create(worker_config, task_def, std::move(iterator), out));
  return out->Restore(reader);
}

FirstComeFirstServedTaskRunner::FirstComeFirstServedTaskRunner(
    std::unique_ptr<TaskIterator> iterator)
    : iterator_(std::move(iterator)), buffer_(/*buffer_size=*/1) {
  // TODO: restore prefetch here once initial checkpointing is fixed.
  // RunPrefetchThread()
}

int64_t FirstComeFirstServedTaskRunner::GetNextElementIndex() {
  auto buffer_el = buffer_.Peek();
  if (buffer_el.ok()) {
    return std::min(buffer_el.ValueOrDie()->element_index, element_index_);
  }
  return element_index_;
}

Status FirstComeFirstServedTaskRunner::Save(SerializationContext* ctx,
                                            IteratorStateWriter* writer)
    TF_LOCKS_EXCLUDED(mu_) {
  VLOG(0) << "DBK: need fcfs lock";
  mutex_lock l(mu_);

  VLOG(0) << "DBK: got fcfs lock";

  auto status = iterator_->Save(ctx, writer);
  VLOG(0) << "DBK: done with iterator saving";

  TF_RETURN_IF_ERROR(writer->WriteScalar(
      FullName("TaskRunner", "FirstComeFirstServed.element_index"),
      this->element_index_));
  VLOG(0) << "DBK: saving element index";

  if (!prefetch_thread_) {
    VLOG(0) << "starting prefetch thread, after first save";
    RunPrefetchThread();
  }
  return status;
}

Status FirstComeFirstServedTaskRunner::Restore(
    VariantTensorDataReader* reader) {
  auto status = reader->ReadScalar(
      FullName("TaskRunner", "FirstComeFirstServed.element_index"),
      &this->element_index_);
  VLOG(0) << "restoring element index in FCSFS: " << element_index_;
  return status;
}

FirstComeFirstServedTaskRunner::~FirstComeFirstServedTaskRunner() { Cancel(); }

Status FirstComeFirstServedTaskRunner::GetNext(const GetElementRequest& req,
                                               GetElementResult& result) {
  /*
    while ((result.components.empty() || result.element_index <
    req.element_index()) && !result.end_of_sequence) {
   */
  DBK_TRACE(" BUFFER_POP_EL");
  TF_ASSIGN_OR_RETURN(result, buffer_.Pop());

  //  VLOG(0) << "(DBK) GetNext in task runner element index: " << (int64_t)
  //  result.element_index << ", components size: " << result.components.size();
  if (result.components.size() > 0) {
    //     Variant x = result.components.at(0);
    //     VLOG(0) << "x: " << x.DebugString();
    //     Variant extracted = x.get<Tensor>()->flat<Variant>()(0);
    //     VLOG(0) << "extracted: " << extracted.DebugString();
    //     CompressedElement *i = extracted.get<CompressedElement>();
    //     VLOG(0) << "iptr " << i;
    //     std::vector<Tensor> out;
    // //    UncompressElement(const CompressedElement &compressed,
    // std::vector<Tensor> *out)
    //     UncompressElement(*i, &out);
    //     VLOG(0) << "vec len after decompress " << out.size();

    //    VLOG(0) << "vec el 0 " << " descibe " <<
    //    out.at(0).SummarizeValue(100);
    VLOG(0) << "Produced element="  //<< out.at(0).SummarizeValue(100, true)
            << "[i=" << result.element_index << ", Task: " << req.task_id()
            << ", EOS: " << result.end_of_sequence << "]";
    // VLOG(0) << "(DBK) componentv2: " <<
    // result.components.at(0).SummarizeValue(100, true) << ", " << i << " for
    // task " << result.element_index;
  }

  if (result.element_index < req.element_index()) {
    DBK_TRACE(" BUFFER_POP_EL_SKIPPED");
  } else {
    DBK_TRACE(" BUFFER_POP_EL_DONE");
  }

  //}

  return Status::OK();
}

Status FirstComeFirstServedTaskRunner::PrefetchFn() {
  while (true) {
    auto status = (buffer_.Push(GetNextFromInputIterator()));

    if (!status.ok()) {
      // the last element doesn't count since we did not manage to enqueue it...
      // element_index_--;
      // VLOG(0) << "FCFS: element index decrease...";
      return status;
    }

    // VLOG(0) << "FCFS prefetch thread fetching...";
  }
  return Status::OK();
}

void FirstComeFirstServedTaskRunner::RunPrefetchThread() {
  auto prefetch_fn = [this] {
    Status status = PrefetchFn();
    if (!status.ok()) {
      buffer_.Cancel(status);
    }
  };
  prefetch_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_fcfs_prefetch_thread",
      prefetch_fn));
}

StatusOr<GetElementResult>
FirstComeFirstServedTaskRunner::GetNextFromInputIterator()
    TF_LOCKS_EXCLUDED(mu_) {
  GetElementResult result;
  std::vector<Tensor> element;
  bool end_of_task;
  result.skip = false;
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_task));
    result.end_of_sequence = end_of_task;
    // VLOG(0) << "DBK: incrementing element index";
    result.element_index = element_index_++;
  }
  if (!end_of_task) {
    result.components = std::move(element);
  }
  return result;
}

void FirstComeFirstServedTaskRunner::Cancel() {
  VLOG(0) << "Cancelling tf.data service FCFS task.";
  buffer_.Cancel(errors::Cancelled("tf.data service FCFS task is cancelled."));
}

model::Model::ModelMetrics FirstComeFirstServedTaskRunner::GetMetrics() {
  return iterator_->GetMetrics();
}

RoundRobinTaskRunner::RoundRobinTaskRunner(
    std::unique_ptr<TaskIterator> iterator, int64_t num_consumers,
    string worker_address)
    : num_consumers_(num_consumers),
      worker_address_(worker_address),
      buffer_(num_consumers_),
      prefetch_thread_(std::move(iterator), num_consumers_) {
  VLOG(1) << "Creating task runner for distributing data round-robin to "
          << num_consumers << " consumers";
}

int64_t RoundRobinTaskRunner::GetNextElementIndex() {
  VLOG(0) << "RoundRobinTaskRunner does not implement GetNextElementIndex. "
             "Terminating!";
  std::terminate();
  return 0;
}

Status RoundRobinTaskRunner::Save(SerializationContext* ctx,
                                  IteratorStateWriter* writer) {
  return prefetch_thread_.iterator_->Save(ctx, writer);
}

Status RoundRobinTaskRunner::Restore(VariantTensorDataReader* reader) {
  return Status::OK();
}

Status RoundRobinTaskRunner::ValidateRequest(const GetElementRequest& req) {
  if (req.consumer_index() < 0 || req.round_index() < 0) {
    return errors::FailedPrecondition(
        "RoundRobinTaskRunner needs to know the consumer index and element "
        "index of each request.");
  }
  if (req.consumer_index() >= num_consumers_) {
    return errors::FailedPrecondition(
        "Requesting data for consumer index ", req.consumer_index(),
        ", but the task is configured for only ", num_consumers_, " consumers");
  }
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareFullRound(int64_t wait_us)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << worker_address_ << ": Preparing full round for round "
          << current_round_;
  // This was the last request to arrive, time to start a new round.
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(wait_us, buffer_));
  round_skipped_ = buffer_.empty();
  new_round_cv_.notify_all();
  return Status::OK();
}

Status RoundRobinTaskRunner::PreparePartialRound()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  VLOG(1) << worker_address_ << ": Starting partial round " << first_round_
          << " for " << requests_[first_round_].size() << " consumers";
  current_round_ = first_round_;
  new_round_cv_.notify_all();
  // Indicates that we need a partial round to get consumers back in sync.
  auto next_round_request = *(requests_[first_round_ + 1].begin()->second);
  if (next_round_request.skipped_previous_round()) {
    VLOG(1) << "Skipping partial round";
    round_skipped_ = true;
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(/*wait_us=*/-1, buffer_));
  round_skipped_ = false;
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareRound(const GetElementRequest& req) {
  mutex_lock l(mu_);
  first_round_ = std::min(first_round_, req.round_index());
  absl::flat_hash_map<int64_t, const GetElementRequest*>& round =
      requests_[req.round_index()];
  round[req.consumer_index()] = &req;
  auto cleanup = gtl::MakeCleanup([&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    requests_[req.round_index()].erase(req.consumer_index());
  });
  if (current_round_ < req.round_index() && round.size() == num_consumers_) {
    current_round_ = req.round_index();
    int64_t wait_us = kWaitBeforeSkipUs;
    if (!req.allow_skip()) {
      wait_us = -1;
    }
    TF_RETURN_IF_ERROR(PrepareFullRound(wait_us));
  }
  if (current_round_ < 0 &&
      requests_[first_round_].size() + requests_[first_round_ + 1].size() ==
          num_consumers_) {
    TF_RETURN_IF_ERROR(PreparePartialRound());
  }
  while (!cancelled_ && current_round_ < req.round_index()) {
    TF_RETURN_IF_ERROR(prefetch_thread_.GetStatus());
    new_round_cv_.wait(l);
  }
  if (current_round_ < req.round_index() && cancelled_) {
    return errors::Cancelled("Worker is shutting down.");
  }
  if (current_round_ != req.round_index()) {
    return errors::FailedPrecondition(
        "Consumer ", req.consumer_index(), " requested data for round ",
        req.round_index(), ", but the current round has already reached ",
        current_round_,
        ". This may indicate that the consumer was restarted with the same job "
        "name.`");
  }
  return prefetch_thread_.GetStatus();
}

Status RoundRobinTaskRunner::GetNext(const GetElementRequest& req,
                                     GetElementResult& result) {
  TF_RETURN_IF_ERROR(ValidateRequest(req));
  result.end_of_sequence = false;
  VLOG(2) << worker_address_ << ": Received request from consumer index "
          << req.consumer_index() << " for round " << req.round_index();
  TF_RETURN_IF_ERROR(PrepareRound(req));
  tf_shared_lock l(mu_);
  result.skip = round_skipped_;
  if (round_skipped_) {
    VLOG(1) << worker_address_ << ": Buffer not ready, skipping round "
            << current_round_ << " for consumer " << req.consumer_index();
    return Status::OK();
  }
  auto& buffer_result = buffer_[req.consumer_index()];
  result.element_index = buffer_result->index;
  std::vector<Tensor> element;
  for (auto& component : buffer_result->components) {
    element.push_back(tensor::DeepCopy(component));
  }
  if (VLOG_IS_ON(2)) {
    int64_t size = 0;
    for (auto& component : element) {
      size += component.TotalBytes();
    }
    VLOG(2) << worker_address_ << ": Returning element " << result.element_index
            << " to consumer " << req.consumer_index() << " for round "
            << req.round_index() << ". element size " << size;
  }
  result.components = std::move(element);
  return Status::OK();
}

model::Model::ModelMetrics RoundRobinTaskRunner::GetMetrics() {
  return prefetch_thread_.iterator_->GetMetrics();
}

void RoundRobinTaskRunner::Cancel() {
  mutex_lock l(mu_);
  cancelled_ = true;
  new_round_cv_.notify_all();
}

PrefetchThread::PrefetchThread(std::unique_ptr<TaskIterator> iterator,
                               int64_t round_size)
    : iterator_(std::move(iterator)), round_size_(round_size) {
  thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "round-robin-prefetch", [&] { Run(); }));
}

PrefetchThread::~PrefetchThread() {
  mutex_lock l(mu_);
  cancelled_ = true;
  cv_.notify_all();
}

void PrefetchThread::Run() {
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && buffer_.size() >= round_size_) {
        cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
    }
    std::vector<Tensor> element;
    bool end_of_sequence;
    Status s = iterator_->GetNext(element, end_of_sequence);
    if (!s.ok()) {
      mutex_lock l(mu_);
      status_ = s;
      cv_.notify_all();
      return;
    }
    if (end_of_sequence) {
      mutex_lock l(mu_);
      status_ = errors::FailedPrecondition(
          "Encountered end of sequence on a round-robin read iterator. "
          "Please ensure that the dataset used for round-robin reading has "
          "infinite cardinality, e.g. by adding a .repeat() transformation "
          "at the end.");
      cv_.notify_all();
      return;
    }
    mutex_lock l(mu_);
    buffer_.push_back(absl::make_unique<Element>(std::move(element), index_++));
    cv_.notify_all();
  }
}

Status PrefetchThread::FillBuffer(int64_t wait_us,
                                  std::vector<std::unique_ptr<Element>>& out) {
  int64_t start_us = Env::Default()->NowMicros();
  out.clear();
  mutex_lock l(mu_);
  while (buffer_.size() < round_size_ && !cancelled_ && status_.ok()) {
    int64_t remaining_us = start_us + wait_us - Env::Default()->NowMicros();
    if (wait_us >= 0 && remaining_us <= 0) {
      break;
    }
    cv_.wait_for(l, std::chrono::microseconds(remaining_us));
  }
  TF_RETURN_IF_ERROR(status_);
  if (cancelled_) {
    return errors::Cancelled("Prefetch thread cancelled");
  }
  if (buffer_.size() < round_size_) {
    DCHECK_GE(wait_us, 0);
    return Status::OK();
  }
  for (auto& elem : buffer_) {
    out.push_back(std::move(elem));
  }
  buffer_.clear();
  cv_.notify_all();
  return Status::OK();
}

Status PrefetchThread::GetStatus() {
  mutex_lock l(mu_);
  return status_;
}
}  // namespace data
}  // namespace tensorflow
