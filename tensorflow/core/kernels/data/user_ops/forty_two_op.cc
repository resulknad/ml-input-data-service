#include "tensorflow/core/kernels/data/user_ops/forty_two_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/data/name_utils.h"

namespace tensorflow {
namespace data {

// @damien-aymon C++, why is redeclaration needed here?
// /* static */ constexpr const char* const FortyTwoDatasetOp::kDatasetType;
constexpr char kInfiniteTake[] = "InfiniteFortyTwo";


FortyTwoDataset::FortyTwoDataset(OpKernelContext* ctx, const DatasetBase* input)
    : DatasetBase(DatasetContext(ctx)), input_(input) {
    input_->Ref();
  }

FortyTwoDataset::FortyTwoDataset(DatasetContext::Params params, const DatasetBase* input)
    : DatasetBase(DatasetContext(std::move(params))), input_(input) {
    input_->Ref();
  }

FortyTwoDataset::~FortyTwoDataset() { input_->Unref(); }

const DataTypeVector& FortyTwoDataset::output_dtypes() const {
  static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
  return *dtypes;
}

const std::vector<PartialTensorShape>& FortyTwoDataset::output_shapes() const {
  std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>(); 
  shapes->push_back(TensorShape()); 
  return *shapes;
}

string FortyTwoDataset::DebugString() const {
  return name_utils::DatasetDebugString(FortyTwoDatasetOp::kDatasetType);
}

Status FortyTwoDataset::CheckExternalState() const {
  return Status::OK();
}

class FortyTwoDataset::InfiniteIterator : public DatasetIterator<FortyTwoDataset> {
  public:
  explicit InfiniteIterator(const Params& params)
      : DatasetIterator<FortyTwoDataset>(params) {}

  Status Initialize(IteratorContext* ctx) override {
    return Status::OK();
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    out_tensors->clear();
    int64 val = 42;
    out_tensors->push_back(Tensor(val));                        
    return Status::OK();
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args), 0);
  }

  /* TODO: Add state saving / restoring logic which uses input_ */
  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override {
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    return Status::OK();
  }
};

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
std::unique_ptr<IteratorBase> FortyTwoDataset::MakeIteratorInternal(
    const string& prefix) const {
  return absl::make_unique<InfiniteIterator>(InfiniteIterator::Params{
        this, name_utils::IteratorPrefix(kInfiniteTake, prefix)});
}


Status FortyTwoDataset::AsGraphDefInternal(SerializationContext* ctx,
                                           DatasetGraphDefBuilder* b,
                                           Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
  TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
  return Status::OK();
}

FortyTwoDatasetOp::FortyTwoDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}


void FortyTwoDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  // Create a new TakeDatasetOp::Dataset, and return it as the output.
  *output = new FortyTwoDataset(ctx, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("FortyTwoDataset").Device(DEVICE_CPU), FortyTwoDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow