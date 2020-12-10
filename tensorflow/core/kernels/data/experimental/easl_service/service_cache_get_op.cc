#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_get_op.h"

#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"


namespace tensorflow {
namespace data {
namespace experimental {
namespace easl{

/* static */ constexpr const char* const ServiceCacheGetOp::kDatasetType;
/* static */ constexpr const char* const ServiceCacheGetOp::kPath;



class ServiceCacheGetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::string& path);

  ~Dataset() override;

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
  class Iterator;

  const tstring path_;

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
  std::unique_ptr<tensorflow::data::easl::service_cache_util::Reader> reader_
  TF_GUARDED_BY(mu_);
};

// -----------------------------------------------------------------------------
// DatasetOp
// -----------------------------------------------------------------------------

ServiceCacheGetOp::ServiceCacheGetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {

  // TODO(damien-aymon) Why does the snapshot op have these attributes, they
  // seem never to be used.
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  //OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ServiceCacheGetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  *output = new ServiceCacheGetOp::Dataset(ctx, path);
}

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Dataset(
    OpKernelContext* ctx, const std::string& path)
    : DatasetBase(DatasetContext(ctx)), path_(path) {}

ServiceCacheGetOp::Dataset::~Dataset() {}

std::unique_ptr<IteratorBase>
ServiceCacheGetOp::Dataset::MakeIteratorInternal(const string& prefix) const {
  return absl::make_unique<Iterator>(
      Iterator::Params{this, absl::StrCat(prefix, "::ServiceCacheGet")});
}

const DataTypeVector& ServiceCacheGetOp::Dataset::output_dtypes() const {
  // TODO (damien-aymon) update this and read from metadata file
  static DataTypeVector* dtypes = new DataTypeVector({DT_INT32});
  return *dtypes;
}

const std::vector<PartialTensorShape>&
ServiceCacheGetOp::Dataset::output_shapes() const {
  // TODO (damien-aymon) update this and read from metadata file!!!
  static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>{TensorShape()};
  return *shapes;
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

  return b->AddDataset(this, /*inputs=*/ {path}, output);
}


// -----------------------------------------------------------------------------
// Iterator
// -----------------------------------------------------------------------------

ServiceCacheGetOp::Dataset::Iterator::Iterator(const Params& params)
    : DatasetIterator<Dataset>(params) {};

Status ServiceCacheGetOp::Dataset::Iterator::Initialize(
    IteratorContext* ctx) {
  reader_ =
      std::make_unique<tensorflow::data::easl::service_cache_util::Reader>(
          dataset()->path_, dataset()->output_dtypes(), ctx->env());

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