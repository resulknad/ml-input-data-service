/*
Fourty-two dataset op
This op should create a dataset which will return infinitly many times
the constant 42.

*/



#ifndef TENSORFLOW_CORE_KERNELS_DATA_USER_OPS_FORTY_TWO_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_USER_OPS_FORTY_TWO_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class FortyTwoDataset : public DatasetBase {
  public:
    FortyTwoDataset(OpKernelContext* ctx, const DatasetBase* input);
    
    FortyTwoDataset(DatasetContext::Params params, const DatasetBase* input);

    ~FortyTwoDataset() override;

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override;

  const std::vector<PartialTensorShape>& output_shapes() const override;

  string DebugString() const override;

  Status CheckExternalState() const override;

 protected:

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;
    
  private:
    class InfiniteIterator;
  const DatasetBase* const input_;

};


class FortyTwoDatasetOp : public UnaryDatasetOpKernel {
  public:
    // @damien-aymon uncomment when needed.
    static constexpr const char* const kDatasetType = "FortyTwo";
    //static constexpr const char* const kInputDataset = "input_dataset";
    //static constexpr const char* const kOutputTypes = "output_types";
    //static constexpr const char* const kOutputShapes = "output_shapes";

    explicit FortyTwoDatasetOp(OpKernelConstruction* ctx);

  protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                     DatasetBase** output) override;
};


}  // namespace data
}  // namespace tensorflow



#endif  // TENSORFLOW_CORE_KERNELS_DATA_USER_OPS_FORTY_TWO_OP_H_



