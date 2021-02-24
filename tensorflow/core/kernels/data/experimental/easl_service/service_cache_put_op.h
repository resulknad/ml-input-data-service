#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_PUT_OP_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_PUT_OP_

// This operation transparently puts the dataset elements into the tf.data
// service cache.
// This op should not be inserted by end-users.

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{


class ServiceCachePutOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ServiceCachePut";
  static constexpr const char* const kPath = "path";
  static constexpr const char* const kParallelism = "parallelism";

  explicit ServiceCachePutOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                     DatasetBase** output) override;

 private:
  class Dataset;

};


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_PUT_OP_