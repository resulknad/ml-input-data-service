#include "tensorflow/core/data/service/easl/cache_utils.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

std::string DatasetPutKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put");
}

std::string DatasetGetKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get");
}

Status DatasetKey(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key){
  if(cache_state.IsDatasetCached(fingerprint, worker_address)){
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_get");
    VLOG(0) << "Use get dataset for fingerprint " << fingerprint
                 << " at worker " << worker_address;
    return Status::OK();
  }

  int64 caching_task;
  TF_RETURN_IF_ERROR(cache_state.GetCachingTaskId(
      fingerprint, worker_address, caching_task));
  if(caching_task == task_id) {
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_put");
    VLOG(0) << "Use put dataset for fingerprint " << fingerprint
                 << " at worker " << worker_address;
    return Status::OK();
  }

  dataset_key =
      absl::StrCat("id_", dataset_id, "_fp_", fingerprint);
  VLOG(0) << "Use standard dataset for fingerprint " << fingerprint
               << " at worker " << worker_address;
  return Status::OK();
}

Status AddPutOperator(const DatasetDef& dataset, DatasetDef& updated_dataset){
  // TODO (damien-aymon) update this to actual implementation.
  updated_dataset = dataset;
  return Status::OK();
}

Status AddGetOperator(const DatasetDef& dataset, DatasetDef& updated_dataset){
  // TODO (damien-aymon) update this to actual implementation.
  updated_dataset = dataset;
  return Status::OK();
}



} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
