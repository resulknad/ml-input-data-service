#include "tensorflow/core/data/service/easl/dispatcher_cache_state.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

CacheState::CacheState() {}

bool CacheState::IsDatasetCached(
    const uint64 fingerprint, const std::string& worker_address) const {
  auto worker_it = is_cached_at_worker_.find(fingerprint);
  if(worker_it == is_cached_at_worker_.end()){
    return false;
  }

  auto is_cached_it = worker_it->second.find(worker_address);
  if(is_cached_it == worker_it->second.end()){
    return false;
  }

  return is_cached_it->second;
}

void CacheState::SetDatasetCached(const uint64 fingerprint,
                                  const std::string& worker_address){
  is_cached_at_worker_[fingerprint][worker_address] = true;
}

Status CacheState::GetCachingTaskId(const uint64 fingerprint,
                                    const std::string &worker_address,
                                    int64 &task_id) const {
  auto worker_it = caching_task_id_for_worker_.find(fingerprint);
  if(worker_it == caching_task_id_for_worker_.end()){
    return errors::NotFound("No task responsible for caching this dataset");
  }

  auto task_it = worker_it->second.find(worker_address);
  if(task_it == worker_it->second.end()) {
    return errors::NotFound("No task responsible for caching this dataset");
  }

  task_id = task_it->second;
  return Status::OK();
}

void CacheState::RegisterCachingTask(const uint64 fingerprint,
                                     const std::string &worker_address,
                                     const int64 task_id) {
  // TODO (damien-aymon) Check if already present??
  caching_task_id_for_worker_[fingerprint][worker_address] = task_id;
}



} // namespace easl
} // namespace data
} // namespace tensorflow

