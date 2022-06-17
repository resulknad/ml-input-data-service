#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"

#include <algorithm>
#include <random>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"
//#include
//"tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_async_writer.h"
//#include
//"tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_async_reader.h"

namespace tensorflow {
namespace data {
namespace easl {
namespace service_cache_util {

namespace {
// Define the max file size as 30 MB
// const int64 kMaxFileSize = 30 * 1e6;
const int64 kMaxFileSize = 125 * 1e6;  // 250MB
}  // namespace

namespace {  // anonymous namespace => declared functions only visible within
             // this file
static constexpr const char* const kCacheLocation = "";

std::string GetFileName(const std::string& shard_directory, int64_t task_id,
                        uint64 writer_id, uint64 file_id,
                        std::string prefix_hash) {
  return io::JoinPath(
      shard_directory,
      strings::Printf("%d_%s_%02llu_%08llu.easl", task_id, prefix_hash.c_str(),
                      static_cast<unsigned long long>(writer_id),
                      static_cast<unsigned long long>(file_id)));
}

}  // namespace

constexpr const char* const kMetadataFilename = "service_cache.metadata";
const int64 kWriterVersion = 2;  // 0 --> ArrowWriter; 2 --> TFRecordWriter
const char kCompression[] = "";  // can be SNAPPY, GZIP, ZLIB, "" for none.

Writer::Writer(Env* env, int64_t task_id, const std::string& target_dir,
               const DataTypeVector& output_dtypes,
               const std::vector<PartialTensorShape>& output_shapes,
               const int writer_count, const int writer_version)
    : env_(env),
      target_dir_(target_dir),
      output_dtypes_(output_dtypes),
      output_shapes_(output_shapes),
      writer_count_(writer_count),
      writer_version_(writer_version),
      task_id_(task_id) {}  // Constructor, store references in object

Writer::~Writer() {}  // ~ Destructor

Status Writer::Initialize() {
  // TODO (damien-aymon) add constant for writer version.

  if (writer_version_ == 0) {  // 0 -> arrow
    // async_writer_ =
    // std::make_unique<arrow_async_writer::ArrowAsyncWriter>(writer_count_);
  } else {
    async_writer_ = std::make_unique<MultiThreadedAsyncWriter>(writer_count_);
  }

  async_writer_->Initialize(
      env_, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0, kCompression,
      writer_version_, output_dtypes_,
      /*done*/
      [this](Status s) {
        // TODO (damien-aymon) check and propagate errors here!
        if (!s.ok()) {
          VLOG(0) << "EASL - writer error: " << s.ToString();
        }
        // LOG(ERROR) << "MultiThreadedAsyncWriter in snapshot writer failed: "
        // << s; mutex_lock l(writer_status_mu_); writer_status_ = s;
        VLOG(3) << "EASL - MultiThreadedAsyncWriter done with status::ok()";
        return;
      },
      task_id_);
  initialized_ = true;
  return Status::OK();
  // return WriteMetadataFile(env_, target_dir_, output_dtypes_,
  // output_shapes_);
}

Status Writer::Write(const std::vector<Tensor>& tensors) {
  async_writer_->Write(tensors);
  // TODO (damien-aymon) check for errors in the async writer
  return Status::OK();
}

Status Writer::Close() {
  if (initialized_) {
    // Will call the destructor and block until done writing.
    async_writer_->SignalEOF();
    async_writer_.reset();
  }
  // TODO(damien-aymon) check status in the async writer.
  return Status::OK();
}

Status Writer::WriteMetadataFile(
    Env* env, const std::string& path, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes) {
  experimental::CacheMetadataRecord metadata;
  metadata.set_creation_timestamp(EnvTime::NowMicros());
  metadata.set_version(kWriterVersion);
  for (const auto& output_dtype : output_dtypes) {
    metadata.add_dtype(output_dtype);
  }
  for (const auto& output_shape : output_shapes) {
    TensorShapeProto* shape_proto = metadata.add_tensor_shape();
    output_shape.AsProto(shape_proto);
  }
  metadata.set_num_writers(writer_count_);

  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(target_dir_));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

// -----------------------------------------------------------------------------
// MultiThreadedAsyncWriter
// -----------------------------------------------------------------------------

MultiThreadedAsyncWriter::~MultiThreadedAsyncWriter() {
  VLOG(0) << "destroying multi threaded async writer";
}
MultiThreadedAsyncWriter::MultiThreadedAsyncWriter(const int writer_count)
    : writer_count_(writer_count), prefix_hash_(GeneratePrefixHash()) {}

std::string MultiThreadedAsyncWriter::GeneratePrefixHash() {
  std::string str(
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
      "klmnopqrstuvwxyz");
  std::random_device rd;
  std::mt19937 generator(rd());
  std::shuffle(str.begin(), str.end(), generator);
  return str.substr(0, 8);
}

void MultiThreadedAsyncWriter::Initialize(
    Env* env, int64 file_index, const std::string& shard_directory,
    uint64 checkpoint_id, const std::string& compression, int64 version,
    const DataTypeVector& output_types, std::function<void(Status)> done,
    int64_t task_id) {
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      env, ThreadOptions(), absl::StrCat("thread_pool_", file_index),
      writer_count_, false);

  VLOG(3) << "(MultiThreadedAsyncWriter) Starting ThreadPool";
  for (int i = 0; i < writer_count_; ++i) {
    thread_pool_->Schedule([this, env, shard_directory, task_id, checkpoint_id,
                            compression, version, &output_types,
                            done = std::move(done), i] {
      // Note that `done` is not used since it causes a bug here
      WriterThread(env, shard_directory, task_id, i, compression, version,
                   output_types);
    });
  }
  VLOG(3) << "(MultiThreadedAsyncWriter) Finished Starting ThreadPool";
}

void MultiThreadedAsyncWriter::Write(const std::vector<Tensor>& tensors) {
  VLOG(3) << "EASL - Entering Write (Multithreaded Async Writer)";

  mutex_lock l(mu_);

  uint64 element_size = 0;
  for (Tensor t : tensors) {
    element_size += t.TotalBytes();
  }

  if (element_size > bytes_per_row_) {  // will hold true for first call since
                                        // bytes_per_element_ default is 0.
    bytes_per_row_ = element_size;
    first_row_info_set_ = true;
    VLOG(0) << "EASL - set bytes per row: " << bytes_per_row_;
  }
  mu_.Await(Condition(this, &MultiThreadedAsyncWriter::ProducerSpaceAvailable));
  VLOG(3) << "EASL - enough space in queue";
  snapshot_util::ElementOrEOF element;
  element.value = tensors;
  element.size = element_size;
  VLOG(0) << "before enquueing to writer: "
          << element.value.back().SummarizeValue(100);
  deque_.push_back(std::move(element));
  queue_size_bytes_ -= element_size;
}

void MultiThreadedAsyncWriter::SignalEOF() {
  mutex_lock l(mu_);
  VLOG(0) << "signaling EOF in MultiThreadedAsyncWriter";

  for (int i = 0; i < writer_count_; ++i) {
    snapshot_util::ElementOrEOF be;
    be.end_of_sequence = true;
    deque_.push_back(std::move(be));
  }
  VLOG(0) << "waiting for threadpool to close";
  thread_pool_->AsEigenThreadPool()->Cancel();
  VLOG(0) << "threadpool closed";
}

void MultiThreadedAsyncWriter::Consume(snapshot_util::ElementOrEOF* be) {
  mutex_lock l(mu_);
  mu_.Await(
      tensorflow::Condition(this, &MultiThreadedAsyncWriter::ElementAvailable));
  *be = deque_.front();
  deque_.pop_front();

  if (!be->end_of_sequence) {
    queue_size_bytes_ -= be->size;
  }
}

bool MultiThreadedAsyncWriter::ProducerSpaceAvailable() {
  return (deque_.size() * bytes_per_row_) < producer_threshold_;
}

bool MultiThreadedAsyncWriter::ElementAvailable() { return !deque_.empty(); }

Status MultiThreadedAsyncWriter::WriterThread(
    Env* env, const std::string& shard_directory, int64_t task_id,
    uint64 writer_id, const std::string& compression, int64 version,
    DataTypeVector output_types) {
  // TODO (damien-aymon) Push this to the specific writers, so that we can make
  // the async writer more general (e.g. different file system, gs://, etc...)
  uint64 file_size = 0;
  uint64 file_index = 0;
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));
  VLOG(3) << "(Writer_" << writer_id << ") Created Directory "
          << shard_directory;

  std::unique_ptr<snapshot_util::Writer> writer;

  TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
      env,
      GetFileName(shard_directory, task_id, writer_id, file_index,
                  prefix_hash_),
      compression, version, std::move(output_types), &writer));

  int count = 0;
  VLOG(3) << "(Writer_" << writer_id << ") Starting to write ";

  while (true) {
    snapshot_util::ElementOrEOF be;
    Consume(&be);

    VLOG(3) << "(Writer_" << writer_id << ") Read - " << be.end_of_sequence
            << " - Total: " << ++count;
    if (be.end_of_sequence) {
      VLOG(0) << "(Writer_" << writer_id << ") closing...";
      writer->Close();
      VLOG(0) << "(Writer_" << writer_id << ") Closed w/ total read " << count;
      break;
    }

    VLOG(0) << "Writer " << writer_id << " wrote el_id "
            << be.value.back().SummarizeValue(100);
    TF_RETURN_IF_ERROR(writer->WriteTensors(be.value));

    // If the current file exceeded size limits, close it and open another
    file_size += be.size;
    if (file_size > kMaxFileSize) {
      writer->Close();
      VLOG(0) << "(Writer_" << writer_id << ") Closed file with " << file_size
              << " bytes written.";
      TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
          env,
          GetFileName(shard_directory, task_id, writer_id, ++file_index,
                      prefix_hash_),
          compression, version, std::move(output_types), &writer));
      count = file_size = 0;
    }
  }
  return Status::OK();
}

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

Status Reader::Initialize() {
  if (reader_version_ == 0) {  // 0 -> arrow
    VLOG(3) << "(Reader) Making use of the Arrow version of the reader";
    //    async_reader_ =
    //    std::make_unique<arrow_async_reader::ArrowAsyncReader>(
    //      env_, split_provider_, target_dir_, output_dtypes_, output_shapes_,
    //      reader_count_);
  } else {
    VLOG(0) << "(Reader) Making use of the non-Arrow version of the reader";
    async_reader_ = std::make_unique<MultiThreadedAsyncReader>(
        env_, split_provider_, target_dir_, output_dtypes_, output_shapes_,
        reader_count_, deterministic_, element_index_);
  }

  return async_reader_->Initialize();
}

Status Reader::Read(std::vector<Tensor>*& read_tensors, bool* end_of_sequence) {
  return async_reader_->Read(read_tensors, end_of_sequence);
}

void Reader::Close() {
  VLOG(0) << "EASL - (Reader::Close) async_reader closing (bool) "
          << async_reader_.operator bool();
  if (async_reader_) {
    async_reader_->Close();
    async_reader_.reset();
  }
}

// -----------------------------------------------------------------------------
// MultiThreadedAsyncReader (Base Class for ArrowAsyncReader)
// -----------------------------------------------------------------------------

/*MultiThreadedAsyncReader::MultiThreadedAsyncReader(Env *env,
  const std::string &target_dir, const DataTypeVector &output_dtypes,
  const std::vector<PartialTensorShape> &output_shapes, const int reader_count)
    : env_(env), split_provider_(nullptr), output_dtypes_(output_dtypes),
    output_shapes_(output_shapes), target_dir_(target_dir),
    reader_count_(reader_count), num_readers_done_(0) {}*/

Status MultiThreadedAsyncReader::Initialize() {
  // Don't use metadata file at the moment...
  // TF_RETURN_IF_ERROR(ReadAndParseMetadataFile());
  if (split_provider_ == nullptr) {
    return errors::FailedPrecondition(
        "EASL - The split_provider is null, cannot initialize the reader");
  }

  // Find all the files of this dataset
  std::vector<string> files;
  TF_CHECK_OK(
      env_->GetMatchingPaths(io::JoinPath(target_dir_, "*\\.easl"), &files));
  file_count_ = files.size();

  std::sort(files.begin(), files.end());

  {
    mutex_lock l(mu_);
    for (const auto& f : files) {
      file_names_.push_back(f);
    }
  }

  VLOG(0) << "file_names_ :";
  for (auto file : file_names_) {
    VLOG(0) << file;
  }

  reader_count_ = std::max(reader_count_, (int)file_names_.size());
  // Spawn the threadpool, and start reading from the files
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      env_, ThreadOptions(), absl::StrCat("reader_thread_pool", reader_count_),
      reader_count_, false);

  if (deterministic_) {
    output_dtypes_.push_back(DT_INT64);
    output_shapes_.push_back(TensorShape({}));
  }

  VLOG(0) << "(Reader) Starting ThreadPool";
  running_readers_ = reader_count_;
  VLOG(0) << "(Reader) with reader count" << reader_count_;
  for (int i = 0; i < reader_count_; ++i) {
    thread_pool_->Schedule([this, i] {
      ReaderThread(env_, i, kWriterVersion, output_dtypes_, output_shapes_);
    });
  }
  VLOG(0) << "(Reader) Finished Starting ThreadPool";
}

void MultiThreadedAsyncReader::Consume(string* s, bool* end_of_sequence) {
  mutex_lock l(mu_);

  // We should be running in distributed epoch mode
  Tensor split;
  VLOG(0) << "(Consume) In split_provider consume";
  // Void function --> can't use TF_RETURN_IF_ERROR
  Status status = split_provider_->GetNext(&split, end_of_sequence);
  VLOG(0) << "(DBK): after split_provider get";
  if (!status.ok()) {
    VLOG(0) << "EASL - (MultiThreadedAsyncReader::Consume) split provider "
               "returned non-ok status"
            << status;
    *end_of_sequence = true;
    *s = "";
  }
  if (*end_of_sequence) {
    VLOG(0) << "EASL - (MultiThreadedAsyncReader::Consume) split provider "
               "reached eos.";
    *s = "";
  } else {
    VLOG(0) << "(DBK): before trying to return stff";
    int64 file_idx = split.scalar<int64>()();
    *s = file_names_[file_idx];
    VLOG(0) << "(Consume) Got index " << file_idx << " which is file " << s;
  }
}

bool MultiThreadedAsyncReader::ProducerSpaceAvailable() {
  VLOG(3) << "ProducerSpaceAvailable, checking: deque_.size(): "
          << deque_.size() << " queue size bytes " << queue_size_bytes_
          << " bytes_per_element_ " << bytes_per_element_
          << "producer_threshold " << producer_threshold_;
  if (cancelled_ || end_of_sequence_) {
    return true;
  } else {
    return queue_size_bytes_ + bytes_per_element_ < producer_threshold_;
    // return (deque_.size() * bytes_per_element_) < producer_threshold_;
  }
}

void MultiThreadedAsyncReader::Add(
    std::vector<Tensor>& tensors,
    std::pair<int64_t, string> element_identifier) {
  auto element_index = element_identifier.first;
  VLOG(0) << "EASL - entering read - Add with element_index " << element_index;
  mutex_lock l(mu_add_);

  uint64 element_size = 0;
  for (Tensor t : tensors) {
    element_size += t.TotalBytes();
  }

  if (element_size >
      bytes_per_element_) {  // will hold true for first call since
                             // bytes_per_element_ default is 0.
    bytes_per_element_ = element_size;
    first_row_info_set_ = true;
    VLOG(0) << "EASL - set bytes per row: " << bytes_per_element_;
  }

  // now we need to wait until
  // 1. all readers have read the next element
  // 2. this reader here has the minimum element id amongst all of them
  blocked_readers_++;

  VLOG(0) << "blocked_readers " << blocked_readers_
          << " and reader_count: " << running_readers_;

  element_index_avail_.emplace(element_identifier);
  // if we do not have the element with the smallest index,
  // we need to wake up all the other readers (which will include the one with
  // smallest element index)
  if (blocked_readers_ == running_readers_ &&
      element_identifier != element_index_avail_.top()) {
    mu_add_cv_.notify_all();
  }
  while (!ProducerSpaceAvailable() || (blocked_readers_ != running_readers_) ||
         (element_index_avail_.top() != element_identifier)) {
    VLOG(0) << "Waiting because " << element_index_avail_.top().first << ", "
            << element_index_avail_.top().second
            << " != " << element_identifier.first << ", "
            << element_identifier.second
            << " or blocked_readers_=" << blocked_readers_
            << " != " << running_readers_;
    mu_add_cv_.wait(l);
  }

  element_index_avail_.pop();
  blocked_readers_--;
  //  mu_add_.Await(Condition(this,
  //    &MultiThreadedAsyncReader::ProducerSpaceAvailable));

  queue_size_bytes_ += element_size;

  if (cancelled_) {
    return;
  }

  VLOG(0) << "Add could write to queue with element index " << element_index
          << " was waiting for " << element_index_;

  snapshot_util::ElementOrEOF element;
  element.value = tensors;
  element.size = element_size;
  element.end_of_sequence = false;
  deque_.push_back(std::move(element));
  element_index_++;
  mu_add_cv_.notify_all();
}

void MultiThreadedAsyncReader::ReaderDone() {
  snapshot_util::ElementOrEOF element;
  element.end_of_sequence = true;

  VLOG(0) << "ReaderDone trying to write to queue";
  blocked_readers_++;
  mutex_lock l(mu_add_);
  mu_add_.Await(
      Condition(this, &MultiThreadedAsyncReader::ProducerSpaceAvailable));
  VLOG(0) << "ReaderDone could write to queue";
  blocked_readers_--;

  if (cancelled_) {
    return;
  }

  deque_.push_back(std::move(element));
}

Status MultiThreadedAsyncReader::ReaderThread(
    Env* env, uint64 writer_id, int64 version, DataTypeVector output_types,
    std::vector<PartialTensorShape> output_shapes) {
  tensorflow::profiler::TraceMe activity(
      "EASLReaderThread", tensorflow::profiler::TraceMeLevel::kVerbose);

  bool end_of_sequence = false;

  while (!end_of_sequence) {
    std::string file_path;
    VLOG(0) << "EASL - (Reader) getting file";
    Consume(&file_path, &end_of_sequence);
    VLOG(0) << "(Reader_" << writer_id << ") Got file " << file_path;

    if (!end_of_sequence) {
      VLOG(0) << "(Reader_" << writer_id << ") Reading file " << file_path;

      std::unique_ptr<snapshot_util::Reader> reader;

      snapshot_util::Reader::Create(env, file_path, io::compression::kNone,
                                    version, output_types, &reader);

      VLOG(0) << "(Reader_" << writer_id << ") Starting to read file "
              << file_path;
      auto base_name = io::Basename(file_path);
      int64 count = 0;
      bool eof = false;
      while (!eof) {
        {
          mutex_lock l(mu_add_);
          if (cancelled_) {
            VLOG(0) << "EASLReaderThread closed for cancelled reader.";
            running_readers_--;
            mu_add_cv_.notify_all();
            return Status::OK();
          }
        }
        std::vector<Tensor> tensors;
        Status s = reader->ReadTensors(&tensors);
        if (s != Status::OK()) {
          eof = true;  // can't break because of TFRecordReader.
          VLOG(0) << "Internal error in TFRecordReader. " << s.ToString();
        } else {
          if (!tensors.empty()) {
            int64_t el_indx = 0;
            if (deterministic_) {
              VLOG(0) << "el_indx: " << tensors.back().SummarizeValue(100);
              el_indx = (int64_t)((Tensor)tensors.back()).flat<int64_t>()(0);
              tensors.pop_back();
              VLOG(0) << "right after reading: (" << writer_id
                      << ") with index " << el_indx;
            } else {
              VLOG(0) << "not in deterministic mode";
            }
            Add(tensors, std::pair<int64_t, string>(el_indx, base_name));
          } else {
            VLOG(0) << "tf record reader is empty";
          }
        }
      }
      VLOG(0) << "(Reader_" << writer_id << ") Finished reading file "
              << file_path << " with " << count << " elements.";
    }
  }

  ReaderDone();

  VLOG(3) << "(Reader_" << writer_id << ") Finishing reading task";
  running_readers_--;
  mu_add_cv_.notify_all();
  return Status::OK();
}

Status MultiThreadedAsyncReader::Read(std::vector<Tensor>*& read_tensors,
                                      bool* end_of_sequence) {
  mutex_lock l(mu_add_);
  if (end_of_sequence_ || cancelled_) {
    *end_of_sequence = true;
    return Status::OK();
  }

  VLOG(3) << "(Reader) Task is getting invoked... Reading ";
  while (true) {
    VLOG(3) << "Read - waiting on mutex condition";
    mu_add_.Await(tensorflow::Condition(
        this, &MultiThreadedAsyncReader::ElementAvailable));

    if (cancelled_ || end_of_sequence_) {
      VLOG(0) << "Read - function got woken up for eos/cancelled";
      return Status::OK();
    }
    VLOG(3) << "Read - can now get an element from queue";
    if (!deque_.empty()) {
      snapshot_util::ElementOrEOF& element = deque_.front();

      if (element.end_of_sequence) {
        VLOG(3) << "End of sequence element, checking num_readers_done_";
        VLOG(0) << "num_readers_done_" << num_readers_done_ << "; reader_count_"
                << reader_count_ << "; running_readers_:" << running_readers_;
        num_readers_done_++;
        deque_.pop_front();

        if (num_readers_done_ == reader_count_) {
          end_of_sequence_ = true;
          *end_of_sequence = true;
          VLOG(3) << "(Reader) End of sequence reached, returning empty.";
          return Status::OK();
        }
      } else {
        VLOG(3) << "Something to read in the queue...";
        uint64 element_size = 0;

        // remove element index again
        //        VLOG(0) << "Got element index " <<
        //        element.value.back().SummarizeValue(100);
        //        element.value.pop_back();
        for (auto tensor : element.value) {
          read_tensors->push_back(tensor);
        }

        queue_size_bytes_ -= element.size;
        deque_.pop_front();
        *end_of_sequence = false;
        return Status::OK();
      }
    }
    // Readers are not done, waiting on data...
    VLOG(3) << "(Reader) Task could not read, waiting... ";
  }
}

void MultiThreadedAsyncReader::Close() {
  mutex_lock l(mu_add_);
  cancelled_ = true;
}

bool MultiThreadedAsyncReader::ElementAvailable() {
  return !deque_.empty() || end_of_sequence_ || cancelled_;
}

MultiThreadedAsyncReader::~MultiThreadedAsyncReader() {}

Status MultiThreadedAsyncReader::ReadAndParseMetadataFile() {
  string metadata_filename = io::JoinPath(target_dir_, kMetadataFilename);
  TF_RETURN_IF_ERROR(env_->FileExists(metadata_filename));

  experimental::CacheMetadataRecord metadata;
  TF_RETURN_IF_ERROR(ReadBinaryProto(env_, metadata_filename, &metadata));

  cache_file_version_ = metadata.version();

  output_dtypes_ = DataTypeVector();
  for (auto dtype : metadata.dtype()) {
    output_dtypes_.push_back(static_cast<DataType>(dtype));
  }

  output_shapes_ = std::vector<PartialTensorShape>();
  for (auto& shape : metadata.tensor_shape()) {
    output_shapes_.emplace_back(PartialTensorShape(shape));
  }

  return Status::OK();
}

}  // namespace service_cache_util
}  // namespace easl
}  // namespace data
}  // namespace tensorflow
