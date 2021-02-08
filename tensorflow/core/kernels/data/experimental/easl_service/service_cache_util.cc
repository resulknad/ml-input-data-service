#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

constexpr const char* const kMetadataFilename = "service_cache.metadata";
const int kWriterVersion = 2;
const char kCompression[] = "SNAPPY"; // can be SNAPPY, GZIP, ZLIB, "" for none.


Writer::Writer(Env* env,
    const std::string& target_dir, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes)
    : env_(env), target_dir_(target_dir), output_dtypes_(output_dtypes),
    output_shapes_(output_shapes) {}

Writer::~Writer() {}

Status Writer::Initialize(){
  // TODO (damien-aymon) add constant for writer version.
  async_writer_ = std::make_unique<snapshot_util::AsyncWriter>(
      env_, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
      kCompression, kWriterVersion, output_dtypes_,
      /*done*/ [this](Status s){
        // TODO (damien-aymon) check and propagate errors here!
        if (!s.ok()) {
          VLOG(0) << "EASL - writer error: "<< s.ToString();
        }
        //LOG(ERROR) << "AsyncWriter in snapshot writer failed: " << s;
        //mutex_lock l(writer_status_mu_);
        //writer_status_ = s;
        return;
      }
  );

  return WriteMetadataFile(env_, target_dir_, output_dtypes_, output_shapes_);
}

Status Writer::Write(const std::vector<Tensor>& tensors){
  async_writer_->Write(tensors);
  // TODO (damien-aymon) check for errors in the async writer
  return Status::OK();
}

Status Writer::Close(){
  // Will call the destructor and block until done writing.
  async_writer_->SignalEOF();
  async_writer_.reset();

  // TODO(damien-aymon) check status in the async writer.
  return Status::OK();
}

Status Writer::WriteMetadataFile(
    Env* env, const std::string& path, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes){
  experimental::CacheMetadataRecord metadata;
  metadata.set_creation_timestamp(EnvTime::NowMicros());
  metadata.set_version(kWriterVersion);
  for (const auto& output_dtype : output_dtypes) {
    metadata.add_dtype(output_dtype);
  }
  for (const auto& output_shape : output_shapes){
    TensorShapeProto* shape_proto = metadata.add_tensor_shape();
    output_shape.AsProto(shape_proto);
  }

  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(target_dir_));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

Reader::Reader(Env *env,
               const std::string &target_dir,
               const DataTypeVector& output_dtypes)
    : target_dir_(target_dir), env_(env), output_dtypes_(output_dtypes) {
  // TODO (damien-aymon) add constant for writer version.

}

Status Reader::Initialize() {

  // Read metadata first:
  // TODO (damien-aymon) not really useful anymore until more info in there
  TF_RETURN_IF_ERROR(ReadAndParseMetadataFile());

  std::string filename = io::JoinPath(target_dir_,
      strings::Printf("%08llu.snapshot",
          static_cast<unsigned long long>(0)));

  return snapshot_util::Reader::Create(
      env_, filename, kCompression,
      /*version*/ cache_file_version_, output_dtypes_, &reader_);
}

Status Reader::Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence) {
  *end_of_sequence = false;
  Status s = reader_->ReadTensors(read_tensors);
  if (!errors::IsOutOfRange(s)) {
    return s;
  }
      //Status status = AdvanceToNextFile(ctx->env());
      /*if (errors::IsNotFound(status)) {
        *end_of_sequence = true;
        return Status::OK();
      } else {
        return status;
      }
    }*/
  *end_of_sequence = true;
  return Status::OK();
}

Reader::~Reader(){}

Status Reader::ReadAndParseMetadataFile() {
  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env_->FileExists(metadata_filename));

  experimental::CacheMetadataRecord metadata;
  TF_RETURN_IF_ERROR(ReadBinaryProto(env_, metadata_filename, &metadata));

  cache_file_version_ = metadata.version();

  output_dtypes_ = DataTypeVector();
  for(auto dtype : metadata.dtype()){
    output_dtypes_.push_back(static_cast<DataType>(dtype));
  }

  output_shapes_ = std::vector<PartialTensorShape>();
  for(auto shape : metadata.tensor_shape()){
    output_shapes_.push_back(PartialTensorShape(shape));
  }

  return Status::OK();
}


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow