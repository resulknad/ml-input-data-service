#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_GET_OP_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_GET_OP_

// This operation transparently puts the dataset elements into the tf.data
// service cache.
// This op should not be inserted by end-users.

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace easl {

class ServiceCacheGetOp : public DatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "ServiceCacheGet";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char *const kPath = "path";
  static constexpr const char* const kParallelism = "parallelism";

  explicit ServiceCacheGetOp(OpKernelConstruction *ctx);

  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override;

 private:
  class Dataset;

  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_GET_OP_