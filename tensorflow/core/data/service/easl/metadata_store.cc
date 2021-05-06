#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

MetadataStore::MetadataStore() : metadata_() {}

Status MetadataStore::GetJobMetrics(int64 job_id, 
  std::shared_ptr<JobMetrics> metrics) const {
  auto it = metadata_.find(job_id);
  if (it == metadata_.end()) {
    return errors::NotFound("Job with id ", job_id, " does not have metrics");
  }
  metrics = it->second;
  return Status::OK();
}

} // namespace easl
} // namespace data
} // namespace tensorflow
