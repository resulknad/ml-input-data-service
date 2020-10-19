#include "tensorflow/core/kernels/data/user_ops/forty_two_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"

namespace tensorflow {
namespace data {

// @damien-aymon C++, why is redeclaration needed here?
/* static */ constexpr const char* const FortyTwoDatasetOp::kDatasetType;


FortyTwoDataset::FortyTwoDataset(OpKernelContext* ctx,
                                 const DatasetBase* input)
    : DatasetBase(DatasetContext(ctx)),
      input_dataset_(input) {
  input_dataset_->Ref(); // increase reference count on dataset object.
}

FortyTwoDataset::FortyTwoDataset(DatasetContext::Params params,
                                 const DatasetBase* input)
    : DatasetBase(DatasetContext(std::move(params))),
      input_dataset_(input) {
  input_dataset_->Ref(); // increase reference count on dataset object.
}

FortyTwoDataset::~FortyTwoDataset() { input_dataset_.Unref(); }

const DataTypeVector& FortyTwoDataset::output_dtypes() const {
  static DataTypeVector* dtypes = new DataTypeVector({DT_UINT8});
  return *dtypes;
}

const std::vector<PartialTensorShape>& FortyTwoDataset::output_shapes() const {
  static TensorShape* shape = new TensorShape({1}})
  return *shape;
}

string FortyTwoDataset::DebugString() const {
  return name_utils::DatasetDebugString(FortyTwoDatasetOp::kDatasetType);
}

Status FortyTwoDataset::CheckExternalState() {
  return Status::OK();
}



namespace {
REGISTER_KERNEL_BUILDER(Name("FortyTwoDataset").Device(DEVICE_CPU), FortyTwoDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow