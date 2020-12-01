#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_

// This operation transparently puts the dataset elements into the tf.data
// service cache.
// This op should not be inserted by end-users.

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

// (damien-aymon)
// Top-level writer that handles writes to the service cache.
// In the future, this class should handle more complex logic on how and where
// to distribute individual elements to cache files.
// For now, this class simply wraps a single snapshot_util::async writer using
// a TFRecordWriter.
class Writer {
  public:
  Writer(const std::string& target_dir);

  virtual Status Write(const std::vector<Tensor>& tensors) = 0;

  virtual ~Writer() {}


  private:
  const std::string target_dir_;

  std::unique_ptr<snapshot_util::AsyncWriter> async_writer_;
};

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#undef // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
