#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_put_op.h"

#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const ServiceCachePutOp::kDatasetType;
/* static */ constexpr const char* const ServiceCachePutOp::kPath;
/* static */ constexpr const char* const ServiceCachePutOp::kCacheFormat;
/* static */ constexpr const char* const ServiceCachePutOp::kCacheCompression;
/* static */ constexpr const char* const ServiceCachePutOp::kParallelism;




class ServiceCachePutOp::Dataset : public DatasetBase {
 public:
   Dataset(OpKernelContext* ctx, const DatasetBase* input,
           const std::string& path, const int32 cache_format,
           const int32 cache_compression, const int32 parallelism);

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
    const tstring path_;
    const int32 cache_format_;
    const int32 cache_compression_;
    const int32 parallelism_;

};

class ServiceCachePutOp::Dataset::Iterator : public DatasetIterator<Dataset> {
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

 private:
  mutex mu_;
  std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  std::unique_ptr<tensorflow::data::easl::service_cache_util::Writer> writer_ TF_GUARDED_BY(mu_); 
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

ServiceCachePutOp::ServiceCachePutOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {

  // TODO(damien-aymon) Why does the snapshot op have these attributes, they
  // seem never to be used.
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ServiceCachePutOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  int32 cache_format;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCacheFormat, &cache_format));

  int32 cache_compression;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCacheCompression, &cache_compression));

  int32 parallelism;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kParallelism, &parallelism));

  *output = new ServiceCachePutOp::Dataset(
      ctx, input, path, cache_format, cache_compression, parallelism);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

ServiceCachePutOp::Dataset::Dataset(
    OpKernelContext* ctx,
    const DatasetBase* input,
    const std::string& path,
    const int32 cache_format,
    const int32 cache_compression,
    const int32 parallelism)
    : DatasetBase(DatasetContext(ctx)), input_(input),
      path_(path), cache_format_(cache_format),
      cache_compression_(cache_compression), parallelism_(parallelism) {
  input_->Ref();
}

ServiceCachePutOp::Dataset::~Dataset() { input_->Unref(); }

std::unique_ptr<IteratorBase>
ServiceCachePutOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  VLOG(0) << "EASL - prefix to put op: " << prefix;
  return absl::make_unique<Iterator>(
      Iterator::Params{this, absl::StrCat(prefix, "::ServiceCachePut")});
}

const DataTypeVector& ServiceCachePutOp::Dataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>&
ServiceCachePutOp::Dataset::output_shapes() const {
  return input_->output_shapes();
}

string ServiceCachePutOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

int64 ServiceCachePutOp::Dataset::Cardinality() const {
  return input_->Cardinality();
}

Status ServiceCachePutOp::Dataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return Status::OK();
}

Status ServiceCachePutOp::Dataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

Status ServiceCachePutOp::Dataset::AsGraphDefInternal(
    SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

  Node* path = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(path_, &path));

  Node* cache_format = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(cache_format_, &cache_format));

  Node* cache_compression = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(cache_compression_, &cache_compression));

  Node* parallelism = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(parallelism_, &parallelism));

  return b->AddDataset(
      this,
      /*inputs=*/
      {input_graph_node, path, cache_format, cache_compression, parallelism},
      output);
}


// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

ServiceCachePutOp::Dataset::Iterator::Iterator(const Params& params)
    : DatasetIterator<Dataset>(params) {};

Status ServiceCachePutOp::Dataset::Iterator::Initialize(
    IteratorContext* ctx) {
  VLOG(0) << "EASL - Initializing ServiceCachePutOp iterator";
  VLOG(0) << "EASL - File format: " << dataset()->cache_format_;
  VLOG(0) << "EASL - parallelism format: " << dataset()->parallelism_;
  // TODO (damien-aymon) compression and file format are available as fields of dataset().
  // Use them for setting up the writers properly.

  writer_ =
      std::make_unique<tensorflow::data::easl::service_cache_util::Writer>(
          ctx->env(), dataset()->path_, dataset()->output_dtypes(),
          dataset()->output_shapes(), dataset()->parallelism_, dataset()->cache_format_);
  writer_->Initialize();

  return dataset()->input_->MakeIterator(
      ctx, this, prefix(), &input_impl_);
}

Status ServiceCachePutOp::Dataset::Iterator::SaveInternal(
    SerializationContext* ctx, IteratorStateWriter* writer) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCachePutOp::Dataset::Iterator::RestoreInternal(
    IteratorContext* ctx, IteratorStateReader* reader) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCachePutOp::Dataset::Iterator::GetNextInternal(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  mutex_lock l(mu_);

  TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
  
  if(*end_of_sequence){
    if(writer_ != nullptr){
      // (damien-aymon) will block until the underlying asyncWriter is done.
      writer_->Close();
      writer_.reset();
    }
    
    return Status::OK();
  }
  std::vector<Tensor> tensors = *out_tensors;
  return writer_->Write(tensors);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ServiceCachePutDataset").Device(DEVICE_CPU),
                        ServiceCachePutOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow