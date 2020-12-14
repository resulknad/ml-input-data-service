#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace cache_utils {

std::string DatasetPutKey(uint64 fingerprint);

std::string DatasetGetKey(uint64 fingerprint);

} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_UTILS_H_
