#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

namespace {

std::string GetFileName(const std::string& shard_directory,
                                uint64 file_id) {
return io::JoinPath(
    shard_directory,
    strings::Printf("%08llu.easl",
                    static_cast<unsigned long long>(file_id)));
}

}

constexpr const char* const kMetadataFilename = "service_cache.metadata";
const int kWriterVersion = 2;

Writer::Writer(Env* env,
    const std::string& target_dir, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes, 
    const int writer_count) : env_(env), target_dir_(target_dir), 
    output_dtypes_(output_dtypes), output_shapes_(output_shapes), 
    writer_count_(writer_count) {}

Writer::~Writer() {}

Status Writer::Initialize(){
  // TODO (damien-aymon) add constant for writer version.
  async_writer_ = std::make_unique<MultiThreadedAsyncWriter>(
      env_, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
      io::compression::kNone, kWriterVersion, output_dtypes_,
      [this](Status s){
        if (!s.ok()) {
          VLOG(0) << "EASL - writer error: "<< s.ToString();
        }
        return;
      },
      writer_count_
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
      strings::Printf("%08llu.easl",
          static_cast<unsigned long long>(0)));

  return snapshot_util::Reader::Create(
      env_, filename,io::compression::kNone,
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

MultiThreadedAsyncWriter::MultiThreadedAsyncWriter(Env* env, int64 file_index,
                         const std::string& shard_directory,
                         uint64 checkpoint_id, const std::string& compression,
                         int64 version, const DataTypeVector& output_types,
                         std::function<void(Status)> done,
                         const int writer_count) : writer_count_(writer_count) {
  thread_pool_ = absl::make_unique<thread::ThreadPool>(env, ThreadOptions(),  
      absl::StrCat("thread_pool_", file_index), writer_count_, false);

  LOG(INFO) << "(MultiThreadedAsyncWriter) Starting ThreadPool"; 
  for (int i = 0; i < writer_count_; ++i) {
    thread_pool_->Schedule(
      [this, env, shard_directory, checkpoint_id, compression, version,
        &output_types, done = std::move(done), i] {
        // Note that `done` is not used since it causes a bug here 
        WriterThread(env, shard_directory, i, compression, version, 
            output_types);
        }
    );
  }
  LOG(INFO) << "(MultiThreadedAsyncWriter) Finished Starting ThreadPool";
}

void MultiThreadedAsyncWriter::Write(const std::vector<Tensor>& tensors) {
  mutex_lock l(mu_);
  snapshot_util::ElementOrEOF element;
  element.value = tensors;
  deque_.push_back(std::move(element));
}

void MultiThreadedAsyncWriter::SignalEOF() {
  mutex_lock l(mu_);
  
  for (int i = 0; i < writer_count_; ++i) {
    snapshot_util::ElementOrEOF be;
    be.end_of_sequence = true;
    deque_.push_back(std::move(be));
  }
}

void MultiThreadedAsyncWriter::Consume(snapshot_util::ElementOrEOF* be) {
  mutex_lock l(mu_);
  mu_.Await(tensorflow::Condition(this, 
      &MultiThreadedAsyncWriter::ElementAvailable));
  *be = deque_.front();
  deque_.pop_front();
}

bool MultiThreadedAsyncWriter::ElementAvailable() { return !deque_.empty(); }

Status MultiThreadedAsyncWriter::WriterThread(Env* env, 
                                 const std::string& shard_directory,
                                 uint64 writer_id,
                                 const std::string& compression, int64 version,
                                 DataTypeVector output_types) { 
  std::unique_ptr<snapshot_util::Writer> writer;
  // TODO (damien-aymon) Push this to the specific writers, so that we can make
  // the async writer more general (e.g. different file system, gs://, etc...)
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));

  LOG(INFO) << "(Writer_" << writer_id << ") Created Dir "; 

  TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
      env, GetFileName(shard_directory, writer_id), 
      compression, version, std::move(output_types), &writer));
  
  int count = 0;
  LOG(INFO) << "(Writer_" << writer_id << ") Starting to write "; 

  while (true) {
    snapshot_util::ElementOrEOF be;
    Consume(&be);

    LOG(INFO) << "(Writer_" << writer_id << ") Read - " 
      << be.end_of_sequence << " - Total: " << ++count;
    if (be.end_of_sequence) {
      LOG(INFO) << "(Writer_" << writer_id << ") Closing w/ total read " << count << "...";
      TF_RETURN_IF_ERROR(writer->Close());
      LOG(INFO) << "(Writer_" << writer_id << ") Closed w/ total read " << count;
      break;
    }

    TF_RETURN_IF_ERROR(writer->WriteTensors(be.value));
  }
  return Status::OK();
}


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow