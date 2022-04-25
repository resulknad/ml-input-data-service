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

#include "tensorflow/core/data/service/worker_impl.h"

#include <memory>
#include <string>
#include <utility>
#include <thread>
#include <chrono>

#include "grpcpp/create_channel.h"
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/auto_shard_rewriter.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/model_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kBytesConsumed[] = "bytes_consumed";
constexpr const char kBytesProduced[] = "bytes_produced";
constexpr const char kNumElements[] = "num_elements";
constexpr const char kInNodeTime[] = "in_node_time";
constexpr const char kInPrefixTime[] = "in_prefix_time";
constexpr const char kBytesPerS[] = "bytes_per_s";
constexpr const char kActiveTime[] = "active_time";
constexpr const char kWorkingTime[] = "working_time";

constexpr int64_t kRetryIntervalMicros = 5 * 1000 * 1000;        // 5 seconds.
constexpr int64_t kDefaultHeartBeatIntervalMs = 30 * 1000;       // 30 seconds.
constexpr int64_t kDefaultDispatcherTimeoutMs = 60 * 60 * 1000;  // 1 hour.

using WorkerConfig = experimental::WorkerConfig;

// Moves the element into the response. If the tensor contains a single
// CompressedElement variant, the move will be zero-copy. Otherwise, the tensor
// data will be serialized as TensorProtos.
Status MoveElementToResponse(std::vector<Tensor>&& element,
                             GetElementResponse& resp) {
  if (element.size() != 1 || element[0].dtype() != DT_VARIANT ||
      !TensorShapeUtils::IsScalar(element[0].shape())) {
    for (const auto& component : element) {
      UncompressedElement* uncompressed = resp.mutable_uncompressed();
      component.AsProtoTensorContent(uncompressed->add_components());
    }
    return Status::OK();
  }
  Variant& variant = element[0].scalar<Variant>()();
  CompressedElement* compressed = variant.get<CompressedElement>();
  if (compressed == nullptr) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a CompressedElement variant tensor, but "
        "it produced ",
        variant.TypeName());
  }
  *resp.mutable_compressed() = *compressed;
  return Status::OK();
}

WorkerConfig ApplyWorkerDefaults(const WorkerConfig& config) {
  WorkerConfig new_config(config);
  if (new_config.heartbeat_interval_ms() == 0) {
    new_config.set_heartbeat_interval_ms(kDefaultHeartBeatIntervalMs);
  }
  if (new_config.dispatcher_timeout_ms() == 0) {
    new_config.set_dispatcher_timeout_ms(kDefaultDispatcherTimeoutMs);
  }
  return new_config;
}
}  // namespace

mutex LocalWorkers::mu_(LINKER_INITIALIZED);
LocalWorkers::AddressToWorkerMap* LocalWorkers::local_workers_ =
    new AddressToWorkerMap();

DataServiceWorkerImpl::DataServiceWorkerImpl(const WorkerConfig& config)
    : config_(ApplyWorkerDefaults(config)), worker_uid_(port::JobUid()) {

  metrics::RecordTFDataServiceWorkerCreated();

  auto checkpoint_env_dir = getenv("DBK_CHECKPOINT_DIR");
  checkpoint_root_ = checkpoint_env_dir ? checkpoint_env_dir : "checkpoints/";
}

DataServiceWorkerImpl::~DataServiceWorkerImpl() {
  mutex_lock l(mu_);
  DBK_TRACE(" END");
  cancelled_ = true;
  task_completion_cv_.notify_one();
  heartbeat_cv_.notify_one();
}

Status DataServiceWorkerImpl::Start(const std::string& worker_address,
                                    const std::string& transfer_address) {
  DBK_TRACE(" START");
  VLOG(0) << "Starting tf.data service worker at address " << worker_address;
  TF_RETURN_IF_ERROR(ValidateWorkerConfig());
  worker_address_ = worker_address;
  transfer_address_ = transfer_address;

  dispatcher_ = absl::make_unique<DataServiceDispatcherClient>(
      config_.dispatcher_address(), config_.protocol());
  TF_RETURN_IF_ERROR(dispatcher_->Initialize());

  Status s = Heartbeat();
  while (!s.ok()) {
    if (!errors::IsUnavailable(s) && !errors::IsAborted(s) &&
        !errors::IsCancelled(s)) {
      VLOG(0) << "Stop trying to register because of " << s;
      return s;
    }
    LOG(WARNING) << "Failed to register with dispatcher at "
                 << config_.dispatcher_address() << ": " << s;
    Env::Default()->SleepForMicroseconds(kRetryIntervalMicros);
    s = Heartbeat();
  }
  LOG(INFO) << "Worker registered with dispatcher running at "
            << config_.dispatcher_address();
  VLOG(0) << "Worker registered with dispatcher running at "
            << config_.dispatcher_address();

  task_completion_thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "data-service-worker-task-completion",
                                  [this]() { TaskCompletionThread(); }));
  heartbeat_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      {}, "data-service-worker-heartbeat", [this]() { HeartbeatThread(); }));
  mutex_lock l(mu_);
  registered_ = true;
  return Status::OK();
}

void DataServiceWorkerImpl::Stop() {
  std::vector<std::shared_ptr<Task>> tasks;
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    for (const auto& entry : tasks_) {
      tasks.push_back(entry.second);
    }
  }
  for (auto& task : tasks) {
    StopTask(*task);
  }
  // At this point there are no outstanding requests in this RPC handler.
  // However, requests successfully returned from this RPC handler may still be
  // in progress within the gRPC server. If we shut down the gRPC server
  // immediately, it could cause these requests to fail, e.g. with broken pipe.
  // To mitigate this, we sleep for some time to give the gRPC server time to
  // complete requests.
  Env::Default()->SleepForMicroseconds(config_.shutdown_quiet_period_ms() *
                                       1000);
}

Status DataServiceWorkerImpl::ValidateWorkerConfig() const {
  const bool any_tag_is_empty = absl::c_any_of(
      config_.worker_tags(),
      [](const std::string& worker_tag) { return worker_tag.empty(); });
  if (any_tag_is_empty) {
    return errors::FailedPrecondition(
        "Worker tags cannot be empty. Got tags {",
        absl::StrJoin(config_.worker_tags().begin(),
                      config_.worker_tags().end(), ", "),
        "}");
  }
  return Status::OK();
}


StatusOr<std::pair<string,int64_t>> DataServiceWorkerImpl::ClosestAvailableCheckpoint(int64_t task_id, int64_t desired_element_id) {
  auto env = tensorflow::Env::Default();

  string checkpoint_dir = io::JoinPath(checkpoint_root_, std::to_string(task_id));

  if (!env->IsDirectory(checkpoint_dir).ok()) {
    return errors::Unavailable("Directory ", checkpoint_dir, " does not exist.");
  }

  std::vector<string> paths {};
  TF_RETURN_IF_ERROR(env->GetMatchingPaths(io::JoinPath(checkpoint_dir, "checkpoint*"), &paths));

  if (paths.size() == 0) {
    return errors::Unavailable("Cannot find anything matching checkpoint pattern in ", checkpoint_dir);
  }
  string max_checkpoint = "";
  int64_t max_checkpoint_element_index = 0;
  // find maximum (newest checkpoint) smaller than the element index the client is looking for
  for (auto path : paths) {
    int64_t checkpoint_element_index = std::stol(path.substr(path.find_last_of(".")+1));
    VLOG(0) << "DBK: " << max_checkpoint << ", " << checkpoint_element_index; 
    if (checkpoint_element_index <= desired_element_id
        && checkpoint_element_index >= max_checkpoint_element_index) {
      max_checkpoint_element_index = checkpoint_element_index;
      max_checkpoint = path;
    }
  }
  if (max_checkpoint.empty()) {
    return errors::Unavailable("Cannot find checkpoint which can produce the index we are looking for.");
  }

  return std::pair<string,int64_t>(max_checkpoint, max_checkpoint_element_index);
}

Status DataServiceWorkerImpl::GetElementResult(
    const GetElementRequest* request, struct GetElementResult* result) {
  Task* task = nullptr;
  bool stop_task = false;
  {
    mutex_lock l(mu_);
    if (cancelled_) {
      return errors::Cancelled("Worker is shutting down");
    }
    if (!registered_) {
      // We need to reject requests until the worker has registered with the
      // dispatcher, so that we don't return NOT_FOUND for tasks that the worker
      // had before preemption.
      VLOG(0) << "(DBK): Worker not yet registered with dispatcher";
      return errors::Unavailable(
          "Worker has not yet registered with dispatcher.");
    }
    auto it = tasks_.find(request->task_id());
    if (it == tasks_.end()) {
      if (deleted_tasks_.contains(request->task_id())) {
        return errors::FailedPrecondition(
            "Got request for local task ", request->task_id(), " of worker ",
            worker_address_, ", which has been deleted. You may be creating ",
            "a duplicate job which has already finished. To fix this, make "
            "sure to create your dataset only once, as opposed to re-creating "
            "it repeatedly inside a loop.");
      }
      if (finished_tasks_.contains(request->task_id())) {
        VLOG(0) << "Task is already finished";
        result->end_of_sequence = true;
        result->skip = false;
        return Status::OK();
      } else {
        // Perhaps the workers hasn't gotten the task from the dispatcher yet.
        // Return Unavailable so that the client knows to continue retrying.
        VLOG(0) << "Task not found (probably not received from dispatcher yet";
        return errors::Unavailable("Task ", request->task_id(), " not found");
      }
      // Perhaps the worker hasn't gotten the task from the dispatcher yet.
      // Return Unavailable so that the client knows to continue retrying.
      return errors::Unavailable("Task ", request->task_id(), " not found");
    }
    task = it->second.get();
    
    UpdateMostRecentElementIndex(request->task_id(), request->element_index());
    TF_RETURN_IF_ERROR(EnsureTaskInitialized(*task));
    
    auto next_available_element = task->task_runner->GetNextElementIndex();
    VLOG(0) << "requesting element index " << request->element_index() << ", can provide >= " << next_available_element;
    if (request->element_index() != next_available_element) {
      auto avail_checkpoint = ClosestAvailableCheckpoint(request->task_id(), request->element_index());

      // check if either
      // 1. the element we can produce is larger than the one we want
      // 2. if we have some checkpoint available with which we could prevent unnecessarily recomputing some of the elements
      if (request->element_index() < task->task_runner->GetNextElementIndex()
          || (avail_checkpoint.ok() && avail_checkpoint.ValueOrDie().second > next_available_element 
            //DBK: the +2 is a "fix" to prevent it recovering from the checkpoint it just made
            // until the checkpointing has been fixed not to "jump" one elmeent due to locking
            // want next checkpoint to be "worth it..."
            + 2)) {
        VLOG(0) << " would like to stop task at this point...";
        VLOG(0) << " next avail element from task runner is " << next_available_element << " while checkpoint val: " << avail_checkpoint.ValueOrDie().second;
        stop_task = true;
      }
    }
  
    if (!stop_task) {
      // must do this within the lock...
      task->outstanding_requests++;
    }
  }
  // moved down here because of non-reentrant lock...
  if (stop_task) {
      VLOG(0) << "Restarting task to allow recovery from earlier checkpoint to statisfy demand " << request->task_id();
      StopTask(*task);
      {
        mutex_lock l(mu_);
        tasks_.erase(request->task_id());
      }
      return errors::Unavailable("Element with ID ",request->element_index(), " not available from current state of task with ID ", request->task_id()); 
  }
  auto cleanup = gtl::MakeCleanup([&] {
    mutex_lock l(mu_);
    task->outstanding_requests--;
    cv_.notify_all();
  });
  TF_RETURN_IF_ERROR(task->task_runner->GetNext(*request, *result));

  if (result->end_of_sequence) {
    mutex_lock l(mu_);
    VLOG(0) << "Reached end_of_sequence for task " << request->task_id();
    VLOG(0) << "Outstanding tasks for this task" << task->outstanding_requests;
    pending_completed_tasks_.insert(request->task_id());
    task_completion_cv_.notify_one();
  }
  heartbeat_cv_.notify_one();
  return Status::OK();
}

Status DataServiceWorkerImpl::ProcessTask(const ProcessTaskRequest* request,
                                          ProcessTaskResponse* response) {
  mutex_lock l(mu_);
  const TaskDef& task = request->task();
  VLOG(3) << "Received request to process task " << task.task_id();
  return ProcessTaskInternal(task);
}

Status DataServiceWorkerImpl::ProcessTaskInternal(const TaskDef& task_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  std::shared_ptr<Task>& task = tasks_[task_def.task_id()];
  if (task) {
    VLOG(1) << "Received request to process already-processed task "
            << task->task_def.task_id();
    return Status::OK();
  }
  task = absl::make_unique<Task>(task_def);
  VLOG(3) << "Began processing for task " << task_def.task_id()
          << " with processing mode "
          << task_def.processing_mode_def().DebugString();
  return Status::OK();
}

Status DataServiceWorkerImpl::GetCheckpointFromDisk(
    DataServiceWorkerImpl::Task& task,
    std::shared_ptr<VariantTensorDataReader>* reader,
    std::vector<std::unique_ptr<VariantTensorData>>* checkpoint_data_scratch,
    bool* checkpoint_available)
    TF_EXCLUSIVE_LOCKS_REQUIRED(task.mu) {
  auto task_id = task.task_def.task_id();


  auto env = tensorflow::Env::Default();
  string checkpoint_dir = io::JoinPath(checkpoint_root_, std::to_string(task_id));


  if (!env->IsDirectory(checkpoint_dir).ok()) {
    *checkpoint_available = false;
    return Status::OK();
  }

  std::vector<string> paths {};
  TF_RETURN_IF_ERROR(env->GetMatchingPaths(io::JoinPath(checkpoint_dir, "checkpoint*"), &paths));


  if (paths.size() == 0) {
    *checkpoint_available = false;
    return Status::OK();
  }


  auto looking_for_index = element_index_for_task_.contains(task_id) ? element_index_for_task_[task_id] : 0; 

  auto avail_checkpoint = ClosestAvailableCheckpoint(task_id, looking_for_index);
  if (!avail_checkpoint.ok()) {
    *checkpoint_available = false;
    VLOG(0) << "Cannot recover s.t. element with index " << looking_for_index << " can be produced. Err: " << avail_checkpoint.status();
    return Status::OK();
  }
  
  auto avail_checkpoint_val = avail_checkpoint.ValueOrDie();
  checkpoint_dir = avail_checkpoint_val.first;
  
  *checkpoint_available = true;
  
  VLOG(0) << "recovering checkpoint with basename " << " (" << checkpoint_dir << ")";
  // iterate from 0 to ... until file does not exist, build the vector of varianttensordata s
  int i = 0;
  string checkpoint_file = io::JoinPath(checkpoint_dir, std::to_string(i));

  std::vector<const VariantTensorData*> checkpoint_data_ptrs {};

  
  while (env->FileExists(checkpoint_file).ok()) {
// Status Env::GetFileSize(const string& fname, uint64* file_size) {
    uint64 fsize;
    TF_RETURN_IF_ERROR(env->GetFileSize(checkpoint_file, &fsize));
    std::unique_ptr<RandomAccessFile> readable_file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(checkpoint_file, &readable_file));
/*
virtual tensorflow::Status Read(uint64 offset, size_t n, StringPiece* result,
                                char* scratch) const = 0;
                                */

    StringPiece res;
    std::unique_ptr<char> scratch = std::unique_ptr<char>(new char[fsize]);
    TF_RETURN_IF_ERROR(readable_file->Read(0, fsize, &res, scratch.get()));
    
    std::unique_ptr<VariantTensorData> vtd(new VariantTensorData());
    if (!vtd->ParseFromString(std::string(res))) {
      return errors::Internal(strings::StrCat("Failed to parse variant tensor from string... ", i));
    }
    checkpoint_data_scratch->push_back(std::move(vtd));
    checkpoint_data_ptrs.push_back(checkpoint_data_scratch->at(checkpoint_data_scratch->size()-1).get());
    // VLOG(0) << checkpoint_data_ptrs.size()-1 <<": " << "DebugString: " << checkpoint_data_ptrs[checkpoint_data_ptrs.size()-1]->DebugString();
    // VLOG(0) << "Metadata: " << checkpoint_data_ptrs[checkpoint_data_ptrs.size()-1]->metadata_;

    i++;
    checkpoint_file = io::JoinPath(checkpoint_dir, std::to_string(i));
  }
  
  reader->reset(new VariantTensorDataReader(checkpoint_data_ptrs));

//  auto writer = std::move(checkpoints.at(task.task_def.task_id()));
//  checkpoints.erase(task.task_def.task_id());
  
//  writer->GetData(&data);
// writer.ReleaseData(&data);
  VLOG(0) << "read some data from file: " << checkpoint_data_ptrs.size() << ", " << checkpoint_data_scratch->size();

  return Status::OK();

}

Status DataServiceWorkerImpl::EnsureTaskInitialized(
    DataServiceWorkerImpl::Task& task) {
  if (task.task_def.worker_address() != worker_address_) {
    return errors::Internal(absl::Substitute(
        "Dispatcher's worker address $0 does not match worker's address $1.",
        task.task_def.worker_address(), worker_address_));
  }

  bool checkpoint_available;
  {
    mutex_lock l(task.mu);
    if (task.initialized) {
      return Status::OK();
    }

    TF_ASSIGN_OR_RETURN(DatasetDef dataset_def, GetDatasetDef(task.task_def));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Dataset> dataset,
                        MakeDataset(dataset_def, task.task_def));

    std::unique_ptr<standalone::Iterator> iterator;

    std::shared_ptr<VariantTensorDataReader> reader; //#(checkpoint_data_ptrs);


    // we must own the data here, it will get discarded after finishing recovery (exiting this function)
    std::vector<std::unique_ptr<VariantTensorData>> checkpoint_data_scratch {};

    std::unique_ptr<StandaloneTaskIterator> task_iterator;
    TF_RETURN_IF_ERROR(GetCheckpointFromDisk(task, &reader, &checkpoint_data_scratch, &checkpoint_available));
    if (checkpoint_available) {
      VLOG(0) << "DBK: Have checkpoint for task!! Trying to read that... godspeed";
      VLOG(0) << "dataset: " << (dataset==nullptr) << ", task.task_def: "; //<< task.task_def.DebugString();
      // possibly need to use MOVE here...
      TF_ASSIGN_OR_RETURN(iterator, MakeDatasetIteratorFromCheckpoint(*dataset, task.task_def, reader.get()));

      task_iterator = absl::make_unique<StandaloneTaskIterator>(
          std::move(dataset), std::move(iterator));
    
    } else {
      VLOG(0) << "DBK: no checkpoint in map. creating iterator as usual...";
      TF_ASSIGN_OR_RETURN(iterator,
                          MakeDatasetIterator(*dataset, task.task_def));
      iterator->SetTaskID(task.task_def.task_id());
      task_iterator = absl::make_unique<StandaloneTaskIterator>(
          std::move(dataset), std::move(iterator));
    }

    TF_RETURN_IF_ERROR(TaskRunner::Create(
        config_, task.task_def, std::move(task_iterator), task.task_runner));

    if (checkpoint_available) {
      task.task_runner->Restore(reader.get());
    }

  }

  // if checkpoint is already available, there is no need to checkpoint again at this point
/*  if (!checkpoint_available) {
    VLOG(0) << "SaveCheckpointToDisk ret val: " << SaveCheckpointToDisk(task);
  }*/
  // make initial checkpoint
  // TODO: start prefetch thread only after checkpoint was done... otherwise we do not get it at element index 0...
  {
    mutex_lock l(task.mu);
    task.initialized = true;
    task.task_checkpointing_thread = absl::WrapUnique(
        Env::Default()->StartThread({}, strings::StrCat("data-service-worker-task", task.task_def.task_id(), "-checkpointing"),
                                    [&task, this]() { TaskCheckpointingThread(task); }));
    VLOG(0) << "Created iterator for task " << task.task_def.task_id();
  }
  return Status::OK();
}

void DataServiceWorkerImpl::TaskCheckpointingThread(Task& task) TF_LOCKS_EXCLUDED(mu_) {
  auto checkpoint_freq_str = getenv("DBK_CHECKPOINT_FREQ_MS");
  int64_t checkpoint_freq_ms = 1000;
  if (checkpoint_freq_str != nullptr) {
    checkpoint_freq_ms = strtoul(checkpoint_freq_str, NULL, 10);
     VLOG(0) << "read checkpoint freq val of " << checkpoint_freq_ms << " from env";
  } else {
     VLOG(0) << "no checkpoint freq val read, used default";
  }
  // make a checkpoint right away...
  int64_t next_checkpoint = Env::Default()->NowMicros();
  VLOG(0) << "Starting chkpting thread for " << task.task_def.task_id();

  while (true) {
    {
      {
        mutex_lock l(task.mu); 
        // wait for either timer or cancelled
        int64_t wait_for = next_checkpoint - (int64_t) Env::Default()->NowMicros();
        if (wait_for < 0) {
          DBK_TRACE(" CHECKPOINT_FREQ_TOO_HIGH");
          VLOG(0) << "(DBK) Can't keep up, checkpointing frequency too high." << wait_for;
          wait_for = 0;
        }

        while (wait_for > 0 || !task.initialized) {
          task.task_checkpointing_cv_.wait_for(l, std::chrono::microseconds(wait_for));
          wait_for = next_checkpoint - (int64_t) Env::Default()->NowMicros();
          if (!task.initialized) {
            VLOG(0) << "task not initialized...";
          }
          if (task.cancelled) {
            VLOG(0) << "Task " << task.task_def.task_id() << " checkpointing thread cancelled";
            return;
          }
        } 
        if (task.cancelled) {
          VLOG(0) << "Task " << task.task_def.task_id() << " checkpointing thread cancelled";
          return;
        }



        next_checkpoint = Env::Default()->NowMicros() + checkpoint_freq_ms*1000; 
      }

      DBK_TRACE(" CHECKPOINT_START");
      auto s = SaveCheckpointToDisk(task);
      VLOG(0) << "SaveCheckpointToDisk ret val: " << s;
      DBK_TRACE(" CHECKPOINT_END");
    }
  }
}

StatusOr<DatasetDef> DataServiceWorkerImpl::GetDatasetDef(
    const TaskDef& task_def) const {
  switch (task_def.dataset_case()) {
    case TaskDef::kDatasetDef:
      return task_def.dataset_def();
    case TaskDef::kPath: {
      DatasetDef def;
      Status s = ReadDatasetDef(task_def.path(), def);
      if (!s.ok()) {
        LOG(INFO) << "Failed to read dataset from " << task_def.path() << ": "
                  << s << ". Falling back to reading from dispatcher.";
        TF_RETURN_IF_ERROR(
            dispatcher_->GetDatasetDef(task_def.dataset_id(), def));
      }
      return def;
    }
    case TaskDef::DATASET_NOT_SET:
      return errors::Internal("Unrecognized dataset case: ",
                              task_def.dataset_case());
  }
}

StatusOr<std::unique_ptr<standalone::Dataset>>
DataServiceWorkerImpl::MakeDataset(const DatasetDef& dataset_def,
                                   const TaskDef& task_def) const {
  TF_ASSIGN_OR_RETURN(AutoShardRewriter auto_shard_rewriter,
                      AutoShardRewriter::Create(task_def));
  // `ApplyAutoShardRewrite` does nothing if auto-sharding is disabled.
  TF_ASSIGN_OR_RETURN(
      GraphDef rewritten_graph,
      auto_shard_rewriter.ApplyAutoShardRewrite(dataset_def.graph()));
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), rewritten_graph, &dataset));
  return dataset;
}

StatusOr<std::unique_ptr<standalone::Iterator>>
DataServiceWorkerImpl::MakeDatasetIteratorFromCheckpoint(standalone::Dataset& dataset,
                                           const TaskDef& task_def,
                                           IteratorStateReader* reader) const {
  std::unique_ptr<standalone::Iterator> iterator;

  if (IsNoShard(task_def.processing_mode_def()) ||
      IsStaticShard(task_def.processing_mode_def())) {
    TF_RETURN_IF_ERROR(dataset.MakeIteratorFromCheckpoint(reader, task_def.task_id(), &iterator));
    return iterator;
  }

  if (IsDynamicShard(task_def.processing_mode_def())) {
    std::vector<std::unique_ptr<SplitProvider>> split_providers;
    split_providers.reserve(task_def.num_split_providers());
    for (int i = 0; i < task_def.num_split_providers(); ++i) {
      split_providers.push_back(absl::make_unique<DataServiceSplitProvider>(
          config_.dispatcher_address(), config_.protocol(), task_def.job_id(),
          i, config_.dispatcher_timeout_ms(), task_def.task_id()));
    }
    /*

    IteratorContext* ctx, const string& output_prefix,
    IteratorStateReader* reader,
    std::unique_ptr<IteratorBase>* iterator) const {
    */
    VLOG(0) << "make iterator from checkpoiint call on dataset";
    TF_RETURN_IF_ERROR(dataset.MakeIteratorFromCheckpoint(std::move(split_providers), task_def.task_id(), reader, &iterator));
    return iterator;
  }

  return errors::InvalidArgument("Unrecognized processing mode: ",
                                 task_def.processing_mode_def().DebugString());
}

StatusOr<std::unique_ptr<standalone::Iterator>>
DataServiceWorkerImpl::MakeDatasetIterator(standalone::Dataset& dataset,
                                           const TaskDef& task_def) const {
  std::unique_ptr<standalone::Iterator> iterator;

  if (IsNoShard(task_def.processing_mode_def()) ||
      IsStaticShard(task_def.processing_mode_def())) {
    TF_RETURN_IF_ERROR(dataset.MakeIterator(task_def.task_id(), &iterator));
    return iterator;
  }

  if (IsDynamicShard(task_def.processing_mode_def())) {
    std::vector<std::unique_ptr<SplitProvider>> split_providers;
    split_providers.reserve(task_def.num_split_providers());
    for (int i = 0; i < task_def.num_split_providers(); ++i) {
      split_providers.push_back(absl::make_unique<DataServiceSplitProvider>(
          config_.dispatcher_address(), config_.protocol(), task_def.job_id(),
          i, config_.dispatcher_timeout_ms(), task_def.task_id()));
    }
    TF_RETURN_IF_ERROR(
        dataset.MakeIterator(std::move(split_providers), task_def.task_id(), &iterator));
    return iterator;
  }

  return errors::InvalidArgument("Unrecognized processing mode: ",
                                 task_def.processing_mode_def().DebugString());
}



Status DataServiceWorkerImpl::SaveCheckpointToDisk(Task& task) { 

  /*
    std::unique_ptr<WritableFile> dump_file;
    string file_name = strings::StrCat(gpu_memory_map_file, "_", Name(), ".",
                                       Env::Default()->NowMicros());
    Status status = Env::Default()->NewWritableFile(file_name, &dump_file);

  FileOutputBuffer(WritableFile* file, size_t buffer_size);
Status WriteVariantTensor(const Tensor& val, FileOutputBuffer* out,
                          size_t* bytes_written, uint32* crc32c) {
*/
  auto task_id = task.task_def.task_id();
  if (!tasks_.count(task_id)) {
    return errors::InvalidArgument("Passed task which is not running anymore to SaveAndDeleteTask");
  }

  // VLOG(0) << "Stopping task" << task_id;
  // StopTask(task);

  VLOG(0) << "Acquiring locks...";


  // since we only read from the task, it's fine to have the shared lock
  // this allows for checkpointing and heartbeats to happen in parallel
  tf_shared_lock l(task.mu);

  // mutex_lock l2(mu_);
  VLOG(0) << "Acquired locks...";

  // SAVING BUSINESS
  VLOG(0) << "DBK: entering SaveAndDeleteTask";
  std::unique_ptr<VariantTensorDataWriter> writer = std::unique_ptr<VariantTensorDataWriter>(new VariantTensorDataWriter());
  std::unique_ptr<SerializationContext> serialization_ctx;
  serialization_ctx = absl::make_unique<SerializationContext>(SerializationContext::Params{});
  VLOG(0) << "DBK: calling iterator_->Save";
  TF_RETURN_IF_ERROR(task.task_runner->Save(serialization_ctx.get(), writer.get()));

  // VLOG(0) << "DBK: calling get data on writer";
  // std::vector<std::unique_ptr<VariantTensorData>> data;
  // writer.ReleaseData(&data);

  //VLOG(0) << "DBK: storing varianttensordata in checkpoints...";
  // checkpoints[task_id] = std::move(writer);
  // VLOG(0) << "DBK: checkpoints size " << checkpoints.size();

  auto env = tensorflow::Env::Default();

  std::vector<const VariantTensorData*> checkpoint_data  {};
  writer->GetData(&checkpoint_data);

  // get element_index of last produced element in checkpoint
  VariantTensorDataReader reader(checkpoint_data);

  Tensor element_index_tensor;
  TF_RETURN_IF_ERROR(reader.ReadTensor(FullName("TaskRunner", "FirstComeFirstServed.element_index"), &element_index_tensor));

  int64_t element_index = element_index_tensor.scalar<int64_t>()();
  string out_dir_tmp = io::JoinPath(checkpoint_root_, std::to_string(task_id), strings::StrCat("tmp.", element_index, Env::Default()->NowMicros()));
  string out_dir = io::JoinPath(checkpoint_root_, std::to_string(task_id), strings::StrCat("checkpoint.", element_index));

  if (env->IsDirectory(out_dir).ok()) {
    VLOG(0) << "stopping with checkpointing (there already is a checkpoint)";
    return Status::OK();
  }
  if (env->IsDirectory(out_dir_tmp).ok()) {
    VLOG(0) << "deleting checkpoint ( there was one before) " << out_dir_tmp;
    int64_t undeleted_dirs, undeleted_files;
    TF_RETURN_IF_ERROR(env->DeleteRecursively(out_dir_tmp, &undeleted_dirs, &undeleted_files)); 
  }
  env->RecursivelyCreateDir(out_dir_tmp);
  
  for (int i=0; i<checkpoint_data.size(); i++) {
    auto str_data = checkpoint_data.at(i)->SerializeAsString();
    // VLOG(0) << "DebugString: " << checkpoint_data[i]->DebugString();
    // VLOG(0) << "Metadata: " << checkpoint_data[i]->metadata_;

    string file_path = io::JoinPath(out_dir_tmp, std::to_string(i));
    std::unique_ptr<WritableFile> ptr;
    env->NewWritableFile(file_path, &ptr);
    ptr->Append(str_data);
    ptr->Close();
    // VLOG(0) << "Wrote " << file_path;
  }

  
  VLOG(0) << "renaming";
  TF_RETURN_IF_ERROR(env->RenameFile(out_dir_tmp, out_dir));
  VLOG(0) << "post renaming";
  // VLOG(0) << "DBK: erasing task";
  // tasks_.erase(task_id);
  return Status::OK(); 
}

void DataServiceWorkerImpl::StopTask(Task& task) TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(task.mu);
    task.initialized = true;
    task.cancelled = true;
    VLOG(0) << "setting task cancelled to true";
    task.task_checkpointing_cv_.notify_all();
  }
  if (task.task_runner) {
    task.task_runner->Cancel();
  }
  mutex_lock l(mu_);
  while (task.outstanding_requests > 0) {
    cv_.wait(l);
  }
}

int64_t DataServiceWorkerImpl::UpdateMostRecentElementIndex(int64_t task_id, int64_t element_index) {
  mutex_lock l(element_index_for_task_mu_);
//  auto old_index = element_index_for_task_.contains(task_id) ? element_index_for_task_[task_id] : kint64max; 

//  if (old_index > element_index) {
    element_index_for_task_.insert_or_assign(task_id, element_index);
//  }

  VLOG(0) << "(DBK): most recent requested element index for task " << task_id << " is " << element_index_for_task_[task_id];
  return element_index_for_task_[task_id];
}

Status DataServiceWorkerImpl::GetElement(const GetElementRequest* request,
                                         GetElementResponse* response) {
  VLOG(0) << "Received GetElement request for task " << request->task_id();
  struct GetElementResult result;
  TF_RETURN_IF_ERROR(GetElementResult(request, &result));
  response->set_element_index(result.element_index);
  response->set_end_of_sequence(result.end_of_sequence);
  response->set_skip_task(result.skip);
  if (!response->end_of_sequence() && !response->skip_task()) {
    TF_RETURN_IF_ERROR(
        MoveElementToResponse(std::move(result.components), *response));
    // VLOG(0) << "Producing an element for task " << request->task_id();
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::GetWorkerTasks(
    const GetWorkerTasksRequest* request, GetWorkerTasksResponse* response) {
  mutex_lock l(mu_);
  for (const auto& it : tasks_) {
    Task* task = it.second.get();
    TaskInfo* task_info = response->add_tasks();
    task_info->set_worker_address(worker_address_);
    task_info->set_task_id(task->task_def.task_id());
    task_info->set_job_id(task->task_def.job_id());
  }
  return Status::OK();
}

void DataServiceWorkerImpl::TaskCompletionThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && pending_completed_tasks_.empty()) {
        task_completion_cv_.wait(l);
      }
      if (cancelled_) {
        VLOG(3) << "Task completion thread shutting down";
        return;
      }
    }

    // EASL - Send heartbeat for metadata: makes sure the metrics have been sent
    // to the dispatcher at least once before the job gets deleted.
    VLOG(1) << "EASL - calling heartbeat from taskCompletionThread";
    Status s = Heartbeat();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send heartbeat to dispatcher: " << s;
    }

    s = SendTaskUpdates();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send task updates to dispatcher: " << s;
      mutex_lock l(mu_);
      if (!cancelled_) {
        task_completion_cv_.wait_for(
            l, std::chrono::microseconds(kRetryIntervalMicros));
      }
    }
  }
}

Status DataServiceWorkerImpl::SendTaskUpdates() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<TaskProgress> task_progress;
  {
    mutex_lock l(mu_);
    VLOG(3) << "Sending " << pending_completed_tasks_.size()
            << " task updates to dispatcher";
    task_progress.reserve(pending_completed_tasks_.size());
    for (int task_id : pending_completed_tasks_) {
      task_progress.emplace_back();
      task_progress.back().set_task_id(task_id);
      task_progress.back().set_completed(true);
    }
  }

  TF_RETURN_IF_ERROR(dispatcher_->WorkerUpdate(worker_address_, task_progress));
  mutex_lock l(mu_);
  for (const auto& update : task_progress) {
    pending_completed_tasks_.erase(update.task_id());
  }
  VLOG(3) << "Sent " << task_progress.size() << " task updates ";
  return Status::OK();
}

void DataServiceWorkerImpl::HeartbeatThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    int64_t next_heartbeat_micros =
        Env::Default()->NowMicros() + (config_.heartbeat_interval_ms() * 1000);
    {
      VLOG(0) << "Heartbeat thread acquiring lock";
      mutex_lock l(mu_);
      VLOG(0) << "Heartbeat thread acquired lock";
      while (!cancelled_ &&
             Env::Default()->NowMicros() < next_heartbeat_micros) {
        int64_t time_to_wait_micros =
            next_heartbeat_micros - Env::Default()->NowMicros();
        heartbeat_cv_.wait_for(l,
                               std::chrono::microseconds(time_to_wait_micros));
      }
      if (cancelled_) {
        VLOG(0) << "Heartbeat thread shutting down";
        return;
      }
      if (!registered_) {
        VLOG(0) << "Not performing heartbeat; worker is not yet registered";
        continue;
      }
    }
    Status s = Heartbeat();
    if (!s.ok()) {
      VLOG(0) << "Failed to send heartbeat to dispatcher: " << s;
      LOG(WARNING) << "Failed to send heartbeat to dispatcher: " << s;
    }
  }
}

Status DataServiceWorkerImpl::Heartbeat() TF_LOCKS_EXCLUDED(mu_) {
//  VLOG(0) << "(DBK) Worker preparing to send hearbeat";
  std::vector<int64_t> current_tasks;
  string tasks_desc;
  absl::flat_hash_map<int64, model::Model::ModelMetrics> tasks_metrics;
  {
    VLOG(0) << "(DataServiceWorkerImpl::Heartbeat) Starting heartbeat" << std::flush;
    mutex_lock l(mu_);
    VLOG(0) << "(DataServiceWorkerImpl::Heartbeat) Acquired lock" << std::flush;
    for (const auto& task : tasks_) {
      current_tasks.push_back(task.first);
      tasks_desc.append(std::to_string(task.first) + " (Job: " + std::to_string(task.second->task_def.job_id()) + "), ");
      // Get the metrics

      // as long as we only READ here, we can allow heartbeats and checkpoints to occur in parallel
      // any writes to task_metrics will need the exclusive lock, thus no race condition possible
      tf_shared_lock l(task.second->mu);
      if (task.second->initialized) {
        VLOG(0) << "Getting metrics in heartbeat";
        VLOG(0) << "worker heartbeat - task.outstanding_requests: " << task.second->outstanding_requests;

        auto metrics = task.second->task_runner->GetMetrics();
        if (metrics) {
          tasks_metrics[task.first] = metrics;
        }
      } else {
        VLOG(0) << "Not getting metrics in heartbeat";
      }
    }
  }

  VLOG(0)<<"(DBK): Worker HB, tasks: " << tasks_desc;

  WorkerHeartbeatRequest request;
  request.set_worker_address(worker_address_);
  request.set_transfer_address(transfer_address_);
  *request.mutable_worker_tags() = config_.worker_tags();
  request.set_worker_uid(worker_uid_);
  *request.mutable_current_tasks() = {current_tasks.begin(),
                                      current_tasks.end()};

  // Add the metrics
  for (auto& task_metrics : tasks_metrics) {
    WorkerHeartbeatRequest::Task* task = request.add_tasks();
    task->set_id(task_metrics.first);
    task->set_last_node_name(
      task_metrics.second->begin()->second.last_node_name());
    task->set_last_tf_node_name(
      task_metrics.second->begin()->second.last_tf_node_name());
    task->set_marker_node_name(
      task_metrics.second->begin()->second.marker_node_name());

    for (auto& node_metrics : *task_metrics.second) {

      WorkerHeartbeatRequest::Task::Node* node = task->add_nodes();
      node->set_name(node_metrics.first);

      // Set the metrics of this node
      auto metrics = node->mutable_metrics();
      (*metrics)[kBytesConsumed] = node_metrics.second.bytes_consumed();
      (*metrics)[kBytesProduced] = node_metrics.second.bytes_produced();
      (*metrics)[kNumElements] = node_metrics.second.num_elements();
      (*metrics)[kInNodeTime] = node_metrics.second.in_node_time();
      (*metrics)[kInPrefixTime] = node_metrics.second.in_prefix_time();
      (*metrics)[kBytesPerS] = node_metrics.second.bytes_per_s();
      (*metrics)[kActiveTime] = node_metrics.second.active_time();
      (*metrics)[kWorkingTime] = node_metrics.second.working_time();
    }
  }

  VLOG(0) << "(DBK) Calling dispatcher WorkerHeartbeat...";
  TF_ASSIGN_OR_RETURN(WorkerHeartbeatResponse response,
                      dispatcher_->WorkerHeartbeat(request));

  std::vector<std::shared_ptr<Task>> tasks_to_delete;
  {
    mutex_lock l(mu_);
    for (const auto& task : response.new_tasks()) {
      VLOG(0) << "Received new task from dispatcher with id " << task.task_id();
      if (deleted_tasks_.contains(task.task_id())) {
        VLOG(0) << "(DBK) Found task id " << task.task_id() << " in deleted tasks";
        continue;
      }
      Status s = ProcessTaskInternal(task);
      VLOG(0) << "(DBK) Processing (new) task id " << task.task_id() << " returned status " << s;
      if (!s.ok() && !errors::IsAlreadyExists(s)) {
        LOG(WARNING) << "Failed to start processing task " << task.task_id()
                     << ": " << s;
      }
    }
    tasks_to_delete.reserve(response.tasks_to_delete_size());
    for (int64_t task_id : response.tasks_to_delete()) {
      VLOG(0) << "Deleting task " << task_id
              << " at the request of the dispatcher";
      if (!tasks_.contains(task_id)) {
        VLOG(0) << "Did not find task locally... ";
        continue;
      }
      tasks_to_delete.push_back(std::move(tasks_[task_id]));
      tasks_.erase(task_id);
      finished_tasks_.insert(task_id);
    }
  }
  for (const auto& task : tasks_to_delete) {
    VLOG(0) << "(DBK) Stopping task " << task->task_def.task_id();
    StopTask(*task);
  }
  VLOG(0) << "(DataServiceWorkerImpl::Heartbeat) Done with heartbeat" << std::flush;
  return Status::OK();
}

void DataServiceWorkerImpl::DeleteLocalTask(const TaskInfo& task_info)
    TF_LOCKS_EXCLUDED(mu_) {
  std::shared_ptr<Task> task;
  {
    mutex_lock l(mu_);
    auto it = tasks_.find(task_info.task_id());
    if (it == tasks_.end() || !it->second) {
      return;
    }
    task = std::move(it->second);
    tasks_.erase(task_info.task_id());
    pending_completed_tasks_.insert(task_info.task_id());
    deleted_tasks_.insert(task_info.task_id());
  }

  VLOG(2) << "Delete local task " << task_info.task_id() << " from worker "
          << worker_address_ << " at the request of the client.";
  StopTask(*task);
}

void LocalWorkers::Add(absl::string_view worker_address,
                       std::shared_ptr<DataServiceWorkerImpl> worker) {
  DCHECK(worker != nullptr) << "Adding a nullptr local worker is disallowed.";
  VLOG(1) << "Register local worker at address " << worker_address;
  mutex_lock l(mu_);
  (*local_workers_)[worker_address] = worker;
}

std::shared_ptr<DataServiceWorkerImpl> LocalWorkers::Get(
    absl::string_view worker_address) {
  tf_shared_lock l(mu_);
  AddressToWorkerMap::const_iterator it = local_workers_->find(worker_address);
  if (it == local_workers_->end()) {
    return nullptr;
  }
  return it->second;
}

bool LocalWorkers::Empty() {
  tf_shared_lock l(mu_);
  return local_workers_->empty();
}

void LocalWorkers::Remove(absl::string_view worker_address) {
  VLOG(1) << "Remove local worker at address " << worker_address;
  mutex_lock l(mu_);
  local_workers_->erase(worker_address);
}

}  // namespace data
}  // namespace tensorflow
