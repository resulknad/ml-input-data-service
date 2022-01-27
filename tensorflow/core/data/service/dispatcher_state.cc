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
#include "tensorflow/core/data/service/dispatcher_state.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

DispatcherState::DispatcherState()
    : worker_index_resolver_(std::vector<std::string>{}) {}

DispatcherState::DispatcherState(
    const experimental::DispatcherConfig& dispatcher_config)
    : worker_index_resolver_(dispatcher_config.worker_addresses()) {}

Status DispatcherState::Apply(const Update& update) {
  switch (update.update_type_case()) {
    case Update::kRegisterDataset:
      RegisterDataset(update.register_dataset());
      break;
    case Update::kRegisterWorker:
      RegisterWorker(update.register_worker());
      break;
    case Update::kCreateJob:
      CreateJob(update.create_job());
      break;
    case Update::kProduceSplit:
      ProduceSplit(update.produce_split());
      break;
    case Update::kAcquireJobClient:
      AcquireJobClient(update.acquire_job_client());
      break;
    case Update::kReleaseJobClient:
      ReleaseJobClient(update.release_job_client());
      break;
    case Update::kGarbageCollectJob:
      GarbageCollectJob(update.garbage_collect_job());
      break;
    case Update::kRemoveTask:
      RemoveTask(update.remove_task());
      break;
    case Update::kCreatePendingTask:
      CreatePendingTask(update.create_pending_task());
      break;
    case Update::kClientHeartbeat:
      ClientHeartbeat(update.client_heartbeat());
      break;
    case Update::kCreateTask:
      CreateTask(update.create_task());
      break;
    case Update::kFinishTask:
      FinishTask(update.finish_task());
      break;
    case Update::kSetElementSpec:
      SetElementSpec(update.set_element_spec());
      break;
    case Update::kReassignFreeWorkers:
      ReassignFreeWorkers();
      break;
    case Update::UPDATE_TYPE_NOT_SET:
      return errors::Internal("Update type not set.");
  }

  return Status::OK();
}

void DispatcherState::RegisterDataset(
    const RegisterDatasetUpdate& register_dataset) {
  int64_t id = register_dataset.dataset_id();
  int64_t fingerprint = register_dataset.fingerprint();
  auto dataset = std::make_shared<Dataset>(id, fingerprint);
  DCHECK(!datasets_by_id_.contains(id));
  datasets_by_id_[id] = dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[fingerprint] = dataset;
  next_available_dataset_id_ = std::max(next_available_dataset_id_, id + 1);
}

void DispatcherState::RegisterWorker(
    const RegisterWorkerUpdate& register_worker) {
  std::string address = register_worker.worker_address();
  DCHECK(!workers_.contains(address));
  DCHECK(!avail_workers_.contains(address));
  workers_[address] = std::make_shared<Worker>(register_worker);
  avail_workers_[address] = std::make_shared<Worker>(register_worker);
  tasks_by_worker_[address] =
      absl::flat_hash_map<int64_t, std::shared_ptr<Task>>();
  jobs_by_worker_[address] =
      absl::flat_hash_map<int64_t, std::shared_ptr<Job>>();

  worker_index_resolver_.AddWorker(address);
}

void DispatcherState::CreateJob(const CreateJobUpdate& create_job) {
  int64_t job_id = create_job.job_id();
  absl::optional<NamedJobKey> named_job_key;
  if (create_job.has_named_job_key()) {
    named_job_key.emplace(create_job.named_job_key().name(),
                          create_job.named_job_key().index());
  }
  absl::optional<int64_t> num_consumers;
  if (create_job.optional_num_consumers_case() ==
      CreateJobUpdate::kNumConsumers) {
    num_consumers = create_job.num_consumers();
  }
  auto job = std::make_shared<Job>(job_id, create_job.dataset_id(),
                                   create_job.processing_mode_def(),
                                   create_job.num_split_providers(),
                                   named_job_key, num_consumers, create_job.target_workers(),
                                   create_job.job_type(), create_job.worker_count(),
                                   create_job.if_use_local_workers());

  for (auto worker: create_job.local_workers()) {
    VLOG(1) << "EASL-MUYU (DispatcherState::CreateJob): worker " << worker;
    job->local_workers.insert(worker);
  }

  DCHECK(!jobs_.contains(job_id));
  jobs_[job_id] = job;
  tasks_by_job_[job_id] = std::vector<std::shared_ptr<Task>>();
  if (named_job_key.has_value()) {
    DCHECK(!named_jobs_.contains(named_job_key.value()) ||
           named_jobs_[named_job_key.value()]->garbage_collected);
    named_jobs_[named_job_key.value()] = job;
  }
  next_available_job_id_ = std::max(next_available_job_id_, job_id + 1);
}

void DispatcherState::ProduceSplit(const ProduceSplitUpdate& produce_split) {
  std::shared_ptr<Job> job = jobs_[produce_split.job_id()];
  DCHECK(job->distributed_epoch_state.has_value());
  DistributedEpochState& state = job->distributed_epoch_state.value();
  int64_t provider_index = produce_split.split_provider_index();
  DCHECK_EQ(produce_split.repetition(), state.repetitions[provider_index]);
  if (produce_split.finished()) {
    state.repetitions[provider_index]++;
    state.indices[provider_index] = 0;
    return;
  }
  state.indices[provider_index]++;
}

void DispatcherState::AcquireJobClient(
    const AcquireJobClientUpdate& acquire_job_client) {
  int64_t job_client_id = acquire_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job);
  job = jobs_[acquire_job_client.job_id()];
  DCHECK(job);
  job->num_clients++;
  next_available_job_client_id_ =
      std::max(next_available_job_client_id_, job_client_id + 1);
}

void DispatcherState::ReleaseJobClient(
    const ReleaseJobClientUpdate& release_job_client) {
  int64_t job_client_id = release_job_client.job_client_id();
  std::shared_ptr<Job>& job = jobs_for_client_ids_[job_client_id];
  DCHECK(job);
  job->num_clients--;
  DCHECK_GE(job->num_clients, 0);
  job->last_client_released_micros = release_job_client.time_micros();
  jobs_for_client_ids_.erase(job_client_id);
}

void DispatcherState::GarbageCollectJob(
    const GarbageCollectJobUpdate& garbage_collect_job) {
  int64_t job_id = garbage_collect_job.job_id();
  for (auto& task : tasks_by_job_[job_id]) {
    task->finished = true;
    tasks_by_worker_[task->worker_address].erase(task->task_id);
  }
  jobs_[job_id]->finished = true;
  jobs_[job_id]->garbage_collected = true;

  // EASL - Update available workers.
  for (auto& worker : workers_by_job_[job_id]) {
    VLOG(0) << "(GarbageCollectJob) Releasing worker at address " << worker->address
            << " for job " << job_id;
    avail_workers_[worker->address] = worker;
    jobs_by_worker_[worker->address].erase(job_id);
  }
  workers_by_job_[job_id].clear();
}

void DispatcherState::RemoveTask(const RemoveTaskUpdate& remove_task) {
  std::shared_ptr<Task>& task = tasks_[remove_task.task_id()];
  DCHECK(task);
  task->removed = true;
  auto& tasks_for_job = tasks_by_job_[task->job->job_id];
  for (auto it = tasks_for_job.begin(); it != tasks_for_job.end(); ++it) {
    if ((*it)->task_id == task->task_id) {
      tasks_for_job.erase(it);
      break;
    }
  }
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  tasks_.erase(task->task_id);
  VLOG(1) << "Removed task " << remove_task.task_id() << " from worker "
          << task->worker_address;
}

void DispatcherState::CreatePendingTask(
    const CreatePendingTaskUpdate& create_pending_task) {
  int64_t task_id = create_pending_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_pending_task.job_id()];
  DCHECK_NE(job, nullptr);
  task = std::make_shared<Task>(create_pending_task, job);
  job->pending_tasks.emplace(task, create_pending_task.starting_round());
  tasks_by_worker_[create_pending_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::ClientHeartbeat(
    const ClientHeartbeatUpdate& client_heartbeat) {
  int64_t job_client_id = client_heartbeat.job_client_id();
  auto& job = jobs_for_client_ids_[job_client_id];
  DCHECK(!job->pending_tasks.empty());
  auto& task = job->pending_tasks.front();
  if (client_heartbeat.has_task_rejected()) {
    task.failures++;
    task.ready_consumers.clear();
    task.target_round = client_heartbeat.task_rejected().new_target_round();
  }
  if (client_heartbeat.task_accepted()) {
    task.ready_consumers.insert(job_client_id);
    if (task.ready_consumers.size() == job->num_consumers.value()) {
      VLOG(1) << "Promoting task " << task.task->task_id
              << " from pending to active";
      task.task->starting_round = task.target_round;
      tasks_by_job_[job->job_id].push_back(task.task);
      job->pending_tasks.pop();
    }
  }
}

void DispatcherState::CreateTask(const CreateTaskUpdate& create_task) {
  int64_t task_id = create_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  auto& job = jobs_[create_task.job_id()];
  DCHECK_NE(job, nullptr);
  task = std::make_shared<Task>(create_task, job);
  tasks_by_job_[create_task.job_id()].push_back(task);
  tasks_by_worker_[create_task.worker_address()][task->task_id] = task;
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::FinishTask(const FinishTaskUpdate& finish_task) {
  VLOG(2) << "Marking task " << finish_task.task_id() << " as finished";
  int64_t task_id = finish_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK(task != nullptr);
  task->finished = true;
  tasks_by_worker_[task->worker_address].erase(task->task_id);
  bool all_finished = true;
  for (const auto& task_for_job : tasks_by_job_[task->job->job_id]) {
    if (!task_for_job->finished) {
      all_finished = false;
    }
  }
  VLOG(0) << "(FinishTask) Job " << task->job->job_id << " finished: " 
          << all_finished;
  jobs_[task->job->job_id]->finished = all_finished;
  // When a job completes, mark its workers as available
  if (all_finished) {
    for (auto& worker : workers_by_job_[task->job->job_id]) {
      VLOG(0) << "(FinishTask) Releasing worker at address " << worker->address
              << " for job " << task->job->job_id;
      avail_workers_[worker->address] = worker;
      jobs_by_worker_[worker->address].erase(task->job->job_id);
    }
    workers_by_job_[task->job->job_id].clear();
  }
}

void DispatcherState::SetElementSpec(
    const SetElementSpecUpdate& set_element_spec) {
  int64_t dataset_id = set_element_spec.dataset_id();
  std::string element_spec = set_element_spec.element_spec();
  DCHECK(!id_element_spec_info_.contains(dataset_id));
  id_element_spec_info_[dataset_id] = element_spec;
}

Status DispatcherState::GetElementSpec(int64_t dataset_id,
                                       std::string& element_spec) const {
  auto it = id_element_spec_info_.find(dataset_id);
  if (it == id_element_spec_info_.end()) {
    return errors::NotFound("Element_spec with key ", dataset_id, " not found");
  }
  element_spec = it->second;
  return Status::OK();
}

int64_t DispatcherState::NextAvailableDatasetId() const {
  return next_available_dataset_id_;
}

Status DispatcherState::DatasetFromId(
    int64_t id, std::shared_ptr<const Dataset>& dataset) const {
  auto it = datasets_by_id_.find(id);
  if (it == datasets_by_id_.end()) {
    return errors::NotFound("Dataset id ", id, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::DatasetFromFingerprint(
    uint64 fingerprint, std::shared_ptr<const Dataset>& dataset) const {
  auto it = datasets_by_fingerprint_.find(fingerprint);
  if (it == datasets_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }
  dataset = it->second;
  return Status::OK();
}

Status DispatcherState::WorkerFromAddress(
    const std::string& address, std::shared_ptr<const Worker>& worker) const {
  auto it = workers_.find(address);
  if (it == workers_.end()) {
    return errors::NotFound("Worker with address ", address, " not found.");
  }
  worker = it->second;
  return Status::OK();
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ListWorkers() const {
  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(workers_.size());
  for (const auto& it : workers_) {
    workers.push_back(it.second);
  }
  return workers;
}

std::vector<std::shared_ptr<const DispatcherState::Worker>>
DispatcherState::ListAvailableWorkers() const {
  std::vector<std::shared_ptr<const Worker>> workers;
  workers.reserve(avail_workers_.size());
  for (const auto& it : avail_workers_) {
    workers.push_back(it.second);
  }
  return workers;
}

std::vector<std::shared_ptr<DispatcherState::Worker>>
DispatcherState::ReserveWorkers(
    int64 job_id, int64 num_worker_remote_target,
    int64 num_worker_local_target,
    const absl::flat_hash_set<std::string> local_workers) {

  std::vector<std::shared_ptr<Worker>> workers;
  workers.reserve(avail_workers_.size());
  VLOG(0) << "(ReserveWorkers) num_worker_total_avail=" << avail_workers_.size()
          << " num_worker_local_avail="  << local_workers.size()
          << " num_worker_remote_target="  << num_worker_remote_target
          << " num_worker_local_target=" << num_worker_local_target;

//  // DCHECK(num_workers <= avail_workers_.size());
//
//  // If the number of required workers is below those available, we just assign
//  // as many as there are available at this epoch's scheduling time.
//  int64 num_workers = target_num_workers <= 0
//    || target_num_workers > avail_workers_.size() ? avail_workers_.size()
//    : target_num_workers;
//  std::vector<std::shared_ptr<Worker>> workers;
//  workers.reserve(num_workers);
//  VLOG(0) << "(ReserveWorkers) User got " << num_workers << " workers from "
//          << "target " << target_num_workers << " workers";
//  VLOG(0) << "(ReserveWorkers) IF_USE_LOCAL_WORKERS is set to " << if_use_local_workers;
//
//  for (auto worker: local_workers) {
//    VLOG(1) << "EASL-MUYU (ReserveWorkers) local_workers: " << worker;
//  }
//
//  int num_local_workers_available = 0;
//  for (auto it = avail_workers_.begin(); it != avail_workers_.end(); it++) {
//    if (local_workers.count(it->first))
//      num_local_workers_available++;
//  }
//  if (if_use_local_workers && num_local_workers_available == 0) {
//    VLOG(0) << "EASL-MUYU (ReserveWorkers): local worker mode is set "
//               "but no local worker is available, change to default mode";
//    if_use_local_workers = false;
//  }
//
//  if (if_use_local_workers) {
//      VLOG(0) << "EASL-DSL (ReserveWorkers): if_use_local_workers is true, so we will first assign a local worker.";
//      bool found_local_worker=false;
//      for (auto it = avail_workers_.begin(); it != avail_workers_.end(); ) {
//        if (local_workers.count(it->first) != 0) {
//          VLOG(0) << "EASL-DSL (ReserveWorkers): we found the worker "
//                  << it->first <<
//                  " (from list avail_workers_) in the job's local_workers list";
//
//          num_workers--;
//          workers.push_back(it->second);
//          VLOG(0) << "EASL-DSL (ReserveWorkers) Assigning local worker at address "
//                  << it->second->address << " to job " << job_id;
//          workers_by_job_[job_id].push_back(it->second);
//          jobs_by_worker_[it->second->address][job_id] = jobs_[job_id];
//          avail_workers_.erase(it++);
//
//          found_local_worker=true;
//          break;
//        }
//        it++;
//      }
//      if(!found_local_worker) {
//        VLOG(0) << "EASL-DSL (ReserveWorkers): we tried but failed to find a local worker to assign.";
//      }
//  }

//  int num_local_workers_available = 0;
//  for (auto it = avail_workers_.begin(); it != avail_workers_.end(); it++) {
//    if (local_workers.count(it->first))
//      num_local_workers_available++;
//  }
//  if (if_use_local_workers && num_local_workers_available == 0) {
//    VLOG(0) << "EASL-MUYU (ReserveWorkers): local worker mode is set "
//               "but no local worker is available, change to default mode";
//    if_use_local_workers = false;
//  }
//
//  if (if_use_local_workers) {
//      VLOG(0) << "EASL-DSL (ReserveWorkers): if_use_local_workers is true, so we will first assign a local worker.";
//      bool found_local_worker=false;
//      for (auto it = avail_workers_.begin(); it != avail_workers_.end(); ) {
//        if (local_workers.count(it->first) != 0) {
//          VLOG(0) << "EASL-DSL (ReserveWorkers): we found the worker "
//                  << it->first <<
//                  " (from list avail_workers_) in the job's local_workers list";
//
//          num_workers--;
//          workers.push_back(it->second);
//          VLOG(0) << "EASL-DSL (ReserveWorkers) Assigning local worker at address "
//                  << it->second->address << " to job " << job_id;
//          workers_by_job_[job_id].push_back(it->second);
//          jobs_by_worker_[it->second->address][job_id] = jobs_[job_id];
//          avail_workers_.erase(it++);
//
//          found_local_worker=true;
//          break;
//        }
//        it++;
//      }
//      if(!found_local_worker) {
//        VLOG(0) << "EASL-DSL (ReserveWorkers): we tried but failed to find a local worker to assign.";
//      }
//  }


  for (auto it = avail_workers_.begin(); it != avail_workers_.end(); ) {
    bool is_local;
    // is_local = std::count(it->second->tags.begin(), it->second->tags.end(), "COLOCATED");  // Tag based
    is_local = local_workers.count(it->first);
    if (is_local) {
        VLOG(1) << "EASL-DSL (ReserveWorkers) Worker_L: " << it->first;
        if (num_worker_local_target <= 0) {
            it++;
            continue;
        } else {
          num_worker_local_target--;
        }
    } else {
        VLOG(1) << "EASL-DSL (ReserveWorkers) Worker_R: " << it->first;
        if (num_worker_remote_target <= 0) {
            it++;
            continue;
        } else {
            num_worker_remote_target--;
        }
    }
    workers.push_back(it->second);
    VLOG(0) << "(ReserveWorkers) Assigning worker at address " 
            << it->second->address << " to job " << job_id;
    workers_by_job_[job_id].push_back(it->second);
    jobs_by_worker_[it->second->address][job_id] = jobs_[job_id];
    avail_workers_.erase(it++);
  }
  VLOG(0) << "(ReserveWorkers) Number of workers for job " << job_id << " is: "
          << workers_by_job_[job_id].size();
  return workers;
}


// Go through jobs linearly and reassign free workers to jobs that miss workers.
void DispatcherState::ReassignFreeWorkers() {
  auto job_iter = jobs_.begin();
  if(job_iter == jobs_.end()){
    // Went through all jobs, can return
    return;
  }

  for(auto it = avail_workers_.begin(); it != avail_workers_.end(); ){
    // Get a job in need of workers
    std::shared_ptr<Job> job = job_iter->second;
    int64 num_assigned_workers = workers_by_job_[job->job_id].size();
    while (job->finished || num_assigned_workers == job->worker_count){
      job_iter++;
      if(job_iter == jobs_.end()){
        // Went through all jobs, can return
        return;
      }
      job = job_iter->second;
      num_assigned_workers = workers_by_job_[job->job_id].size();
    }
    // Assign one worker to the job
    workers_by_job_[job->job_id].push_back(it->second);
    jobs_by_worker_[it->second->address][job->job_id] = jobs_[job->job_id];


    VLOG(0) << "EASL - (ReassignFreeWorkers) Reassigned worker "
    << it->second->address << " to job " << job->job_id;

    avail_workers_.erase(it++);
  }
}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobs() {
  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  jobs.reserve(jobs_.size());
  for (const auto& it : jobs_) {
    jobs.push_back(it.second);
  }
  return jobs;
}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobsForWorker(const absl::string_view worker_address) {
  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  auto it = jobs_by_worker_.find(worker_address);
  if (it == jobs_by_worker_.end()) {
    VLOG(4) << "Worker at address " << worker_address
            << " is not yet assigned to any jobs.";
  }

  const absl::flat_hash_map<int64, std::shared_ptr<Job>>& worker_jobs =
      it->second;
  jobs.reserve(worker_jobs.size());
  for (const auto& job : worker_jobs) {
    jobs.push_back(job.second);
  }
  return jobs;
}

Status DispatcherState::JobFromId(int64_t id,
                                  std::shared_ptr<const Job>& job) const {
  auto it = jobs_.find(id);
  if (it == jobs_.end()) {
    return errors::NotFound("Job id ", id, " not found");
  }
  job = it->second;
  return Status::OK();
}

Status DispatcherState::NamedJobByKey(NamedJobKey named_job_key,
                                      std::shared_ptr<const Job>& job) const {
  auto it = named_jobs_.find(named_job_key);
  if (it == named_jobs_.end()) {
    return errors::NotFound("Named job key (", named_job_key.name, ", ",
                            named_job_key.index, ") not found");
  }
  job = it->second;
  return Status::OK();
}

int64_t DispatcherState::NextAvailableJobId() const {
  return next_available_job_id_;
}

Status DispatcherState::JobForJobClientId(int64_t job_client_id,
                                          std::shared_ptr<const Job>& job) {
  job = jobs_for_client_ids_[job_client_id];
  if (!job) {
    return errors::NotFound("Job client id not found: ", job_client_id);
  }
  return Status::OK();
}

std::vector<int64_t> DispatcherState::ListActiveClientIds() {
  std::vector<int64_t> ids;
  for (const auto& it : jobs_for_client_ids_) {
    if (it.second && !it.second->finished) {
      ids.push_back(it.first);
    }
  }
  return ids;
}

int64_t DispatcherState::NextAvailableJobClientId() const {
  return next_available_job_client_id_;
}

Status DispatcherState::TaskFromId(int64_t id,
                                   std::shared_ptr<const Task>& task) const {
  auto it = tasks_.find(id);
  if (it == tasks_.end()) {
    return errors::NotFound("Task ", id, " not found");
  }
  task = it->second;
  return Status::OK();
}

Status DispatcherState::TasksForJob(
    int64_t job_id, std::vector<std::shared_ptr<const Task>>& tasks) const {
  auto it = tasks_by_job_.find(job_id);
  if (it == tasks_by_job_.end()) {
    return errors::NotFound("Job ", job_id, " not found");
  }
  tasks.clear();
  tasks.reserve(it->second.size());
  for (const auto& task : it->second) {
    tasks.push_back(task);
  }
  return Status::OK();
}

Status DispatcherState::TasksForWorker(
    absl::string_view worker_address,
    std::vector<std::shared_ptr<const Task>>& tasks) const {
  tasks.clear();
  auto it = tasks_by_worker_.find(worker_address);
  if (it == tasks_by_worker_.end()) {
    return errors::NotFound("Worker ", worker_address, " not found");
  }
  const absl::flat_hash_map<int64_t, std::shared_ptr<Task>>& worker_tasks =
      it->second;
  tasks.reserve(worker_tasks.size());
  for (const auto& task : worker_tasks) {
    tasks.push_back(task.second);
  }
  return Status::OK();
}

int64_t DispatcherState::NextAvailableTaskId() const {
  return next_available_task_id_;
}

Status DispatcherState::ValidateWorker(absl::string_view worker_address) const {
  return worker_index_resolver_.ValidateWorker(worker_address);
}

StatusOr<int64_t> DispatcherState::GetWorkerIndex(
    absl::string_view worker_address) const {
  return worker_index_resolver_.GetWorkerIndex(worker_address);
}

}  // namespace data
}  // namespace tensorflow
