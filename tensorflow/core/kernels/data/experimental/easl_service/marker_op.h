//
// Created by aymond on 16.07.21.
//

#ifndef ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_MARKER_OP_H_
#define ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_MARKER_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

class MarkerOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "MarkerDataset";
  static constexpr const char* const kMarkerNodeType = "marker_type";

  explicit MarkerOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  tstring marker_type_;

};

} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_MARKER_OP_H_
