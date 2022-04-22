#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_put_op.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tstring.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"
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


constexpr char kElementIndex[] = "element_index";
constexpr char kCacheFile[] = "cache_files";


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

  int64_t CardinalityInternal() const override;

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

  ~Iterator();

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
  int64_t element_index_ = 0;
  int64_t task_id_ = 0;
  Env* env_;
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
  VLOG(3) << "EASL - prefix to put op: " << prefix;
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

int64_t ServiceCachePutOp::Dataset::CardinalityInternal() const {
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

ServiceCachePutOp::Dataset::Iterator::~Iterator(){
  if(writer_) {
    // (damien-aymon) will block until the underlying asyncWriter is done.
    writer_->Close();
  }
  writer_.reset();
}

Status ServiceCachePutOp::Dataset::Iterator::Initialize(
    IteratorContext* ctx) {
  task_id_ = ctx->task_id();
  env_ = ctx->env();
  VLOG(0) << "EASL - Initializing ServiceCachePutOp iterator TaskID: " << task_id_;
  VLOG(3) << "EASL - File format: " << dataset()->cache_format_;
  VLOG(3) << "EASL - parallelism format: " << dataset()->parallelism_;
  // TODO (damien-aymon) compression and file format are available as fields of dataset().
  // Use them for setting up the writers properly.


  writer_ =
      std::make_unique<tensorflow::data::easl::service_cache_util::Writer>(
          env_, task_id_, dataset()->path_, dataset()->output_dtypes(),
          dataset()->output_shapes(), dataset()->parallelism_, dataset()->cache_format_);

  return dataset()->input_->MakeIterator(
      ctx, this, prefix(), &input_impl_);
}

Status ServiceCachePutOp::Dataset::Iterator::SaveInternal(
    SerializationContext* ctx, IteratorStateWriter* writer) {

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));

  if(writer_){
    VLOG(0) << "Closing writer...";
    TF_RETURN_IF_ERROR(writer_->Close());
  } else {
    VLOG(0) << "writer already closed";
  }

  // persisting list of files

  // Find all the files of this dataset
  std::vector<string> files;
  TF_RETURN_IF_ERROR(env_->GetMatchingPaths(io::JoinPath(dataset()->path_, absl::StrCat(task_id_, "*\\.easl")),
      &files));

  int64_t fcount = static_cast<int64_t>(files.size());
  TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(absl::StrCat(kCacheFile)), fcount));
  VLOG(0) << "file count written: " << fcount;
  VLOG(0) << "File count:" << files.size();
  for (int i=0; i<files.size(); i++) {
    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(absl::StrCat(kCacheFile, ".", i)), files[i]));
    VLOG(0) << "Writing " << files[i];
  }

  TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementIndex), element_index_));
  VLOG(0) << "saving cache put op";

  VLOG(0) << "Making new writer...";
  if(writer_){
    writer_ =
        std::make_unique<tensorflow::data::easl::service_cache_util::Writer>(
            env_, task_id_, dataset()->path_, dataset()->output_dtypes(),
            dataset()->output_shapes(), dataset()->parallelism_, dataset()->cache_format_);
  }


  return Status::OK();
}

Status ServiceCachePutOp::Dataset::Iterator::RestoreInternal(
    IteratorContext* ctx, IteratorStateReader* reader) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
  int64_t file_count;
  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCacheFile), &file_count));

  absl::flat_hash_set<string> files {};
  for (uint64_t i=0; i<file_count; i++) {
    auto key = full_name(absl::StrCat(kCacheFile, ".", i));
    tstring file_name;
    reader->ReadScalar(key, &file_name);
    files.insert(std::string(file_name));
    VLOG(0) << "Just read " << std::string(file_name) << " from storage (file count: " << file_count << ")";
  }

  std::vector<string> files_on_fs;
  TF_RETURN_IF_ERROR(env_->GetMatchingPaths(io::JoinPath(dataset()->path_, absl::StrCat(task_id_, "*\\.easl")),
      &files_on_fs));
  
  for (auto file_name : files_on_fs) {
    if (!files.contains(file_name)) {
      VLOG(0) << "Deleting " << file_name;
      TF_RETURN_IF_ERROR(env_->DeleteFile(file_name));
    } else {
      files.erase(file_name);
    }
  }

  if (files.size() != 0) {
    return errors::FailedPrecondition("Cache dir is not consistent with what checkpoint expects: (amongst others)", *files.begin(), files.size());
  }

  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kElementIndex), &element_index_));
  VLOG(0) << "restoring cache put op: (element_index_: " << element_index_ << ")";
  return Status::OK();
}

Status ServiceCachePutOp::Dataset::Iterator::GetNextInternal(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  mutex_lock l(mu_);
  //VLOG(0) << "ServiceCachePutOp - Get next enter";

  if(writer_ && !writer_->Initialized()){
    TF_RETURN_IF_ERROR(writer_->Initialize());
  }


  TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors, end_of_sequence));

  if(*end_of_sequence){
    if(writer_){
      // (damien-aymon) will block until the underlying asyncWriter is done.
      writer_->Close();
      writer_.reset();
      if(writer_){
        VLOG(0) << "Writer reset, writer_.bool(): true";
      } else {
        VLOG(0) << "Writer reset, writer_.bool(): false";
        if(writer_ != nullptr){
          VLOG(0) << "but writer != nullptr";
        }
      }
    }
    return Status::OK();
  }

  std::vector<Tensor> tensors = *out_tensors;
  VLOG(0) << "before pushing el index tensor " << element_index_;
  tensors.push_back(Tensor(element_index_));
  TF_RETURN_IF_ERROR(writer_->Write(tensors));
  VLOG(0) << "after writing el index tensor " << element_index_;
  element_index_++; 
  return Status::OK();
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ServiceCachePutDataset").Device(DEVICE_CPU),
                        ServiceCachePutOp);
}  // namespace


} // namespace easl
} // namespace experimental
} // namespace data
} // namespace tensorflow
