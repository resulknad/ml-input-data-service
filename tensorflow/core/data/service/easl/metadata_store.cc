#include "tensorflow/core/data/service/easl/metadata_store.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace easl {

MetadataStore::MetadataStore() {}

Status MetadataStore::UpdateMetadata(const uint64 &fingerprint,
                                     const int64 &update) {
  metadata_by_fingerprint_[fingerprint] = update;
  return Status::OK();
}

Status MetadataStore::MetadataFromFingerprint(
    uint64 fingerprint,
    std::shared_ptr<int64> &metadata) const {

  auto it = metadata_by_fingerprint_.find(fingerprint);
  if (it == metadata_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }

  *metadata = it->second;
  return Status::OK();
}

} // namespace easl
} // namespace data
} // namespace tensorflow
