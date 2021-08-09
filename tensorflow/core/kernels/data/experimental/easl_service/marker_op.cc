//
// Created by aymond on 16.07.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/marker_op.h"

#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const MarkerOp::kDatasetType;
/* static */ constexpr const char* const MarkerOp::kMarkerNodeType;



class MarkerOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          const std::string& marker_type);

  ~Dataset() override;

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override;

  const std::vector<PartialTensorShape>& output_shapes() const override;

  string DebugString() const override;

  int64 Cardinality() const override;

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override;

  Status CheckExternalState() const override;

 protected:

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class Iterator;

  const DatasetBase* const input_;
  const std::string marker_type_;
};

class MarkerOp::Dataset::Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params& params);

  Status Initialize(IteratorContext* ctx) override;

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override;

 protected:
  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override;

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override;

  std::shared_ptr<model::Node> CreateNode(IteratorContext* ctx, 
    model::Node::Args args) const override;

 private:
  // mutex mu_;
  std::unique_ptr<IteratorBase> input_impl_;
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

MarkerOp::MarkerOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {

  // TODO(damien-aymon) Why does the snapshot op have these attributes, they
  // seem never to be used.
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kMarkerNodeType, &marker_type_));
}

void MarkerOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {

  *output = new Dataset(ctx, input, marker_type_);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

MarkerOp::Dataset::Dataset(
    OpKernelContext* ctx,
    const DatasetBase* input,
    const std::string& marker_type)
    : DatasetBase(DatasetContext(ctx)), input_(input),
      marker_type_(marker_type){
  input_->Ref();
}

MarkerOp::Dataset::~Dataset() { input_->Unref(); }

std::unique_ptr<IteratorBase>
MarkerOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  VLOG(3) << "EASL - prefix to Marker op: " << prefix;
  return absl::make_unique<Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

const DataTypeVector& MarkerOp::Dataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>&
MarkerOp::Dataset::output_shapes() const {
  return input_->output_shapes();
}

string MarkerOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

int64 MarkerOp::Dataset::Cardinality() const {
  return input_->Cardinality();
}

Status MarkerOp::Dataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return Status::OK();
}

Status MarkerOp::Dataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

Status MarkerOp::Dataset::AsGraphDefInternal(
    SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

  AttrValue marker_type;
  b->BuildAttrValue(marker_type_, &marker_type);

  return b->AddDataset(
      this,
      /*inputs=*/
      {input_graph_node},
      /*attr*/{std::make_pair(kMarkerNodeType, marker_type)},
      output);
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

MarkerOp::Dataset::Iterator::Iterator(const Params& params)
    : DatasetIterator<Dataset>(params) {};

Status MarkerOp::Dataset::Iterator::Initialize(IteratorContext* ctx) {
  return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
}

Status MarkerOp::Dataset::Iterator::SaveInternal(
    SerializationContext* ctx, IteratorStateWriter* writer) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status MarkerOp::Dataset::Iterator::RestoreInternal(
    IteratorContext* ctx, IteratorStateReader* reader) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status MarkerOp::Dataset::Iterator::GetNextInternal(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  // mutex_lock l(mu_);
  return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
}

std::shared_ptr<model::Node> 
MarkerOp::Dataset::Iterator::CreateNode(IteratorContext* ctx, 
  model::Node::Args args) const {
  args.name = args.name + ":" + dataset()->marker_type_;
  return model::MakeKnownRatioNode(args, 1);
}

namespace {
  REGISTER_KERNEL_BUILDER(Name("MarkerDataset").Device(DEVICE_CPU), MarkerOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow