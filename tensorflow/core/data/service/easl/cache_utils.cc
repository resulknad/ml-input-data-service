#include "tensorflow/core/data/service/easl/cache_utils.h"

#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

std::string DatasetPutKey(uint64 fingerprint) {
  return absl::StrCat("fp_", fingerprint, "_put");
}

std::string DatasetGetKey(uint64 fingerprint) {
  return absl::StrCat("fp_", fingerprint, "_get");
}

} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
