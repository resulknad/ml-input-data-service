#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_get_op.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/platform/tstring.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const ServiceCacheGetOp::kDatasetType;
/* static */ constexpr const char* const ServiceCacheGetOp::kPath;
/* static */ constexpr const char* const ServiceCacheGetOp::kCacheFormat;
/* static */ constexpr const char* const ServiceCacheGetOp::kCacheCompression;
/* static */ constexpr const char* const ServiceCacheGetOp::kParallelism;



class ServiceCacheGetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::string& path,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          const int32 cache_format, const int32 cache_compression,
          const int32 parallelism);

  ~Dataset() override;

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override;

  const std::vector<PartialTensorShape>& output_shapes() const override;

  string DebugString() const override;

  Status CheckExternalState() const override;

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>* result) 
    const override;

 protected:

  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class Iterator;

  const tstring path_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  const int32 cache_format_;
  const int32 cache_compression_;
  const int32 parallelism_;
  Env* env_;

};

class ServiceCacheGetOp::Dataset::Iterator : public DatasetIterator<Dataset> {
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
  std::shared_ptr<SplitProvider> split_provider_;
  std::unique_ptr<tensorflow::data::easl::service_cache_util::Reader> reader_;
  TF_GUARDED_BY(mu_);
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

ServiceCacheGetOp::ServiceCacheGetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {

  // (damien-aymon)This op does not have access to the original input dataset
  // it replaces. The dtypes and shapes must therefore be set as attributes
  // of this op.
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ServiceCacheGetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  int32 cache_format;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCacheFormat, &cache_format));

  int32 cache_compression;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCacheCompression, &cache_compression));

  int32 parallelism;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kParallelism, &parallelism));

  *output = new ServiceCacheGetOp::Dataset(
      ctx, path, output_dtypes_, output_shapes_,
      cache_format, cache_compression, parallelism);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Dataset(
    OpKernelContext* ctx,
    const std::string& path,
    const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes,
    const int32 cache_format,
    const int32 cache_compression,
    const int32 parallelism)
    : DatasetBase(DatasetContext(ctx)),
    path_(path),
    output_dtypes_(output_dtypes),
    output_shapes_(output_shapes),
    cache_format_(cache_format),
    cache_compression_(cache_compression),
    parallelism_(parallelism),
    env_(ctx->env()) {}

ServiceCacheGetOp::Dataset::~Dataset() {}

std::unique_ptr<IteratorBase>
ServiceCacheGetOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  VLOG(0) << "EASL - prefix to get op: " << prefix;
  return absl::make_unique<Iterator>(
      Iterator::Params{this, absl::StrCat(prefix, "::ServiceCacheGet")});
}

const DataTypeVector& ServiceCacheGetOp::Dataset::output_dtypes() const {
  return output_dtypes_;
}

const std::vector<PartialTensorShape>&
ServiceCacheGetOp::Dataset::output_shapes() const {
  return output_shapes_;
}

string ServiceCacheGetOp::Dataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

Status ServiceCacheGetOp::Dataset::CheckExternalState() const {
  return Status::OK();
}

Status ServiceCacheGetOp::Dataset::AsGraphDefInternal(
    SerializationContext* ctx, DatasetGraphDefBuilder* b, Node** output) const {

  Node* path = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(path_, &path));

  Node* cache_format = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(cache_format_, &cache_format));

  Node* cache_compression = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(cache_compression_, &cache_compression));

  Node* parallelism = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(parallelism_, &parallelism));

  // TODO(damien-aymon) add all fields to graph..)

  return b->AddDataset(
      this,
      /*inputs=*/ {path, cache_format, cache_compression, parallelism},
      output);
}

Status ServiceCacheGetOp::Dataset::MakeSplitProviders(
  std::vector<std::unique_ptr<SplitProvider>>* split_providers) const {
  std::vector<string> files;
  TF_CHECK_OK(env_->GetMatchingPaths(io::JoinPath(path_, "*\\.easl"), &files));
  split_providers->push_back(
    absl::make_unique<IndexSplitProvider>(files.size()));
  return Status::OK();
}

// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Iterator::Iterator(const Params& params)
    : DatasetIterator<Dataset>(params) {};

Status ServiceCacheGetOp::Dataset::Iterator::Initialize(
    IteratorContext* ctx) {
  VLOG(0) << "EASL - Initializing ServiceCacheGet iterator";
  VLOG(0) << "EASL - File format: " << dataset()->cache_format_;
  VLOG(0) << "EASL - Compression format: " << dataset()->cache_compression_;

  // If we're in distributed epoch mode we should have a split provider
  std::shared_ptr<SplitProvider> split_provider_ = nullptr;
  if (!ctx->split_providers().empty()) {
    split_provider_ = ctx->split_providers()[0];
  }

  for(auto dt: dataset()->output_dtypes_){
    VLOG(0) << DataTypeString(dt);
  }
  reader_ =
      std::make_unique<tensorflow::data::easl::service_cache_util::Reader>(
          ctx->env(), split_provider_, dataset()->path_, 
          dataset()->output_dtypes_, dataset()->output_shapes_, dataset()->parallelism_, dataset()->cache_format_);

  return reader_->Initialize();
}

Status ServiceCacheGetOp::Dataset::Iterator::SaveInternal(
    SerializationContext* ctx, IteratorStateWriter* writer) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCacheGetOp::Dataset::Iterator::RestoreInternal(
    IteratorContext* ctx, IteratorStateReader* reader) {
  return errors::Unimplemented("Checkpointing is currently not supported.");
}

Status ServiceCacheGetOp::Dataset::Iterator::GetNextInternal(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  mutex_lock l(mu_);
  auto model = ctx->model();

  return reader_->Read(out_tensors, end_of_sequence);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ServiceCacheGetDataset").Device(DEVICE_CPU),
                        ServiceCacheGetOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow