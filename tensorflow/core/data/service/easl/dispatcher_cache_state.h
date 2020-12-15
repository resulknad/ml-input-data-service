//
// Created by damien-aymon on 14.12.20.
//

#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_

#include "tensorflow/core/lib/core/status.h"
#include "absl/container/flat_hash_map.h"

namespace tensorflow {
namespace data {
namespace easl {

// This class holds the state of the tf.data service cache.
class CacheState {
 public:
  CacheState();
  CacheState(const CacheState &) = delete;
  CacheState &operator=(const CacheState &) = delete;

  bool IsDatasetCached( const uint64 fingerprint,
                        const std::string& worker_address) const;

  void SetDatasetCached(const uint64 fingerprint,
                        const std::string& worker_address);

  // Sets the task_id responsible for caching the dataset with this
  // fingerprint at this worker.
  // Returns an error if the task_id is not found.
  Status GetCachingTaskId(const uint64 fingerprint,
                          const std::string& worker_address,
                          int64& task_id) const;

  void RegisterCachingTask(const uint64 fingerprint,
                           const std::string& worker_address,
                           const int64 task_id);

 private:
  absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, bool>>
      is_cached_at_worker_;

  absl::flat_hash_map<uint64, absl::flat_hash_map<std::string, int64>>
      caching_task_id_for_worker_;



};

} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_DISPATCHER_CACHE_STATE_H_
