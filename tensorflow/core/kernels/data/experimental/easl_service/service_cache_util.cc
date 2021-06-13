#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_async_reader.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_round_robin.h"


namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

namespace {
static constexpr const char *const kCacheLocation = "";

std::string GetFileName(const std::string& shard_directory,
                                uint64 file_id, uint64 split_id = 0) {
  return io::JoinPath(shard_directory, strings::Printf("%08llu.easl",
                      static_cast<unsigned long long>(file_id)));
}

}

constexpr const char* const kMetadataFilename = "service_cache.metadata";
const int64 kWriterVersion = 2; // 0 --> ArrowWriter; 2 --> TFRecordWriter
const char kCompression[] = ""; // can be SNAPPY, GZIP, ZLIB, "" for none.

Writer::Writer(Env* env,
    const std::string& target_dir, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes, 
    const int writer_count, const int writer_version, const int compression) : env_(env), target_dir_(target_dir),
    output_dtypes_(output_dtypes), output_shapes_(output_shapes), 
    writer_count_(writer_count), writer_version_(writer_version), compression_(compression) {}  // Constructor, store references in object

Writer::~Writer()= default;

Status Writer::Initialize(){
  if(writer_version_ == 0) { // 0 -> arrow
//    async_writer_ = std::make_unique<arrow_async_writer::ArrowAsyncWriter>(writer_count_);
  } else if(writer_version_ == 7) {
    async_writer_ = absl::make_unique<arrow_round_robin::ArrowRoundRobinWriter>(writer_count_, MEMORY_THRESHOLD_);

    #ifdef DEBUGGING
    VLOG(0) << "[Writer] Created Round Robin Writer.";
    #endif

  } else {
    async_writer_ = absl::make_unique<TFRecordWriter>(writer_count_, MEMORY_THRESHOLD_);

    #ifdef DEBUGGING
    VLOG(0) << "[Writer] Created TFRecordWriter.";
    #endif
  }

  VLOG(0) << "async_writer nullptr? : " << (async_writer_ == nullptr);
  async_writer_->Initialize(env_, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
                            kCompression, writer_version_, output_dtypes_,
          /*done*/ [this](Status s){
              // TODO (damien-aymon) check and propagate errors here!
              if (!s.ok()) {
                VLOG(0) << "EASL - writer error: "<< s.ToString();
              }
              //LOG(ERROR) << "MultiThreadedAsyncWriter in snapshot writer failed: " << s;
              //mutex_lock l(writer_status_mu_);
              //writer_status_ = s;
              return;
          });

  return Status::OK();
}

Status Writer::Write(const std::vector<Tensor>& tensors){
  async_writer_->Write(tensors);
  return Status::OK();
}

Status Writer::Close(){
  // Will call the destructor and block until done writing.
  async_writer_->SignalEOF();
  async_writer_.reset();
  return Status::OK();
}

// -----------------------------------------------------------------------------
// BoundedMemoryWriter
// -----------------------------------------------------------------------------

BoundedMemoryWriter::BoundedMemoryWriter(const int writer_count, const uint64 memory_threshold) :
        memory_threshold_(memory_threshold), writer_count_(writer_count) {
  bytes_per_row_ = 0;
  bytes_written_ = 0;
  bytes_received_ = 0;
  writers_finished_ = 0;
  first_row_info_set_ = false;
  available_row_capacity_ = 0;

  #ifdef DEBUGGING
  VLOG(0) << "[BoundedMemoryWriter] Constructed BoundedMemoryWriter...";
  #endif

  #ifdef STATS_LOG
  logger_ = absl::make_unique<StatsLogger>();
  #endif
}



void BoundedMemoryWriter::Initialize(Env *env, int64 file_index, const std::string &shard_directory,
        uint64 checkpoint_id, const std::string &compression, int64 version,
        const DataTypeVector &output_types, std::function<void (Status)> done) {

  thread_pool_ = absl::make_unique<thread::ThreadPool>(env, "mem_bounded_threads", writer_count_);

  #ifdef DEBUGGING
  VLOG(0) << "[BoundedMemoryWriter] Finished Creating Threadpool.";
  #endif

  for (int i = 0; i < writer_count_; ++i) {
    thread_pool_->Schedule(
            [this, env, shard_directory, compression, output_types, version, i] {
                // Note that `done` is not used since it causes a bug here
                WriterThread(env, shard_directory, i, compression, output_types, version);
            }
    );
  }

  #ifdef DEBUGGING
  VLOG(0) << "[BoundedMemoryWriter] Finished Initialization";
  #endif
}

Status BoundedMemoryWriter::Write(const std::vector<Tensor> &tensors) {
  #ifdef STATS_LOG
  logger_->WriteInvoked();
  #endif

  if(!first_row_info_set_) {
    for(const Tensor& t : tensors) {
      size_t bytes = t.TotalBytes();
      bytes_per_row_ += bytes;
    }

    FirstRowInfo(tensors);  // inheriting sub classes can extract needed information here
    first_row_info_set_ = true;
    available_row_capacity_ = memory_threshold_ / bytes_per_row_;

    #ifdef DEBUGGING
    VLOG(0) << "[BoundedMemoryWriter] Extracted first row info. mem_thresh: " << memory_threshold_ << "  bpr: " << bytes_per_row_ << ""
                     " available row cap: " << available_row_capacity_;
    #endif

    assert(memory_threshold_ > bytes_per_row_);  // has to hold, otherwise get negative av_row_capacity below.
  }

  bytes_received_ += bytes_per_row_;
  available_row_capacity_--;

  InsertData(tensors);

  // if insert data returns true --> don't have to check if pipeline full
  if(available_row_capacity_ >= 1) {
    #ifdef STATS_LOG
    logger_->WriteReturn();
    #endif
    return Status::OK();
  }


  // check if pipeline full upon receiving next batch -> go to sleep:
  mutex_lock lb(mu_by_);
  if(bytes_received_ - bytes_written_ > memory_threshold_ - bytes_per_row_) {
    #ifdef DEBUGGING
    VLOG(0) << "[BoundedMemoryWriter] Write Pipeline Full, Going to Sleep...";
    #endif

    #ifdef STATS_LOG
    logger_->WriteSleep();
    #endif

    mu_by_.Await(Condition(this,
                &BoundedMemoryWriter::ProducerSpaceAvailable));

    #ifdef STATS_LOG
    logger_->WriteAwake();
    #endif

    #ifdef DEBUGGING
    VLOG(0) << "[BoundedMemoryWriter] write wake up, new capacity in Pipeline...";
    #endif
  }

  // update row capacity available
  uint64 bytes_in_pipeline = bytes_received_ - bytes_written_;
  available_row_capacity_ = (memory_threshold_ - bytes_in_pipeline)  / bytes_per_row_;

  #ifdef STATS_LOG
  logger_->WriteReturn();
  #endif
  return Status::OK();
}

bool BoundedMemoryWriter::ProducerSpaceAvailable() const {
  return bytes_received_ - bytes_written_ <= memory_threshold_ - bytes_per_row_;
}

std::unique_ptr<ElementOrEOF> BoundedMemoryWriter::Consume(int writer_id) {
  mutex_lock l(mu_);

  #ifdef DEBUGGING
  auto before = std::chrono::high_resolution_clock::now();
  #endif

  mu_.Await(tensorflow::Condition(this,
            &BoundedMemoryWriter::ElementAvailable));

  #ifdef DEBUGGING
  auto after = std::chrono::high_resolution_clock::now();
  auto since_last = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
  if(since_last > 1000000000) {
    VLOG(0) << "[Thread " << writer_id << "] was sleeping for " << since_last / 1000000 << "ms to consume queue element";
  }
  #endif

  std::unique_ptr<ElementOrEOF> temp = std::move(deque_.front());
  deque_.pop_front();
  return std::move(temp);
}

bool BoundedMemoryWriter::ElementAvailable() const {
  return !deque_.empty();
}

void BoundedMemoryWriter::SignalEOF() {
  #ifdef DEBUGGING
  VLOG(0) << "[BoundedMemoryWriter] signalling eof";
  #endif

  Cleanup();  // Letting subclasses clean up their data structures before destruction.

  mutex_lock l(mu_);

  for (int i = 0; i < writer_count_; ++i) {
    std::unique_ptr<ElementOrEOF> eof_token = CreateEOFToken();
    deque_.push_back(std::move(eof_token));
  }

  // wait for all writers to finish.
  mu_.Await(tensorflow::Condition(this,
            &BoundedMemoryWriter::AllWritersFinished));

  // print stats summary of all writers
  #ifdef STATS_LOG
  for(int i = 0; i < writer_count_; i++) {
    logger_->PrintStatsSummary(i);
  }
  #endif
}

bool BoundedMemoryWriter::AllWritersFinished() const {
  return writers_finished_ == writer_count_;
}


// -----------------------------------------------------------------------------
// TFRecord Writer
// -----------------------------------------------------------------------------

TFRecordWriter::TFRecordWriter(int writer_count, uint64 memory_threshold) :
          BoundedMemoryWriter(writer_count, memory_threshold){}

void TFRecordWriter::InsertData(const std::vector<Tensor> &tensors) {
  mutex_lock l(mu_);
  std::unique_ptr<RowOrEOF> r_dat = absl::make_unique<RowOrEOF>();
  r_dat->eof = false;
  r_dat->data = tensors;
  deque_.push_back(std::move(r_dat));
}

void TFRecordWriter::FirstRowInfo(const std::vector<Tensor> &tensors) {}  // leave empty

std::unique_ptr<ElementOrEOF> TFRecordWriter::CreateEOFToken() {
  std::unique_ptr<RowOrEOF> r_eof = absl::make_unique<RowOrEOF>();
  r_eof->eof = true;
  return std::move(r_eof);
}

void TFRecordWriter::WriterThread(Env *env, const std::string &shard_directory,
                  int writer_id, const std::string& compression, const DataTypeVector& output_types, int64 version) {

  #ifdef DEBUGGING
  VLOG(0) << "[Thread " << writer_id << "] started running.";
  #endif

  env->RecursivelyCreateDir(shard_directory);

  #ifdef DEBUGGING
  VLOG(0) << "[Thread " << writer_id << "] created shard directory.";
  #endif

  std::unique_ptr<snapshot_util::Writer> writer;

  snapshot_util::Writer::Create(
          env, GetFileName(shard_directory, writer_id),
          "", version, output_types, &writer);

  #ifdef DEBUGGING
  VLOG(0) << "[Thread " << writer_id << "] created TFRecord writer.";
  #endif

  while (true) {
    // parent_be now has ownership over the pointer. When out of scope destructed
    std::unique_ptr<ElementOrEOF> parent_be = Consume(writer_id);
    auto* r_be = dynamic_cast<RowOrEOF*>(parent_be.get());

    if (r_be->eof) {
      #ifdef DEBUGGING
      VLOG(0) << "[Thread " << writer_id << "] closing TFRecord writer...";
      #endif
      writer->Close();
      break;
    }

    BeforeWrite(writer_id);
    // TODO for now: measure time it takes to serialize tensors to string:
    for(Tensor& t : r_be->data) {
      TensorProto proto;
      t.AsProtoTensorContent(& proto);
      auto proto_buffer = new std::string();
      proto.SerializeToString(proto_buffer);
      delete proto_buffer;
    }
    FinishedConversion(writer_id);
    writer->WriteTensors(r_be->data);
    AfterWrite(writer_id);

    mu_by_.lock();
    bytes_written_ += bytes_per_row_;
    mu_by_.unlock();
  }
  WriterReturn(writer_id);
}


// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

Reader::Reader(Env *env, const std::string &target_dir, const DataTypeVector& output_dtypes,
        const std::vector<PartialTensorShape>& output_shapes, const int reader_count, const int reader_version)
        : target_dir_(target_dir), env_(env), output_dtypes_(output_dtypes), reader_count_(reader_count),
        reader_version_(reader_version){}

Status Reader::Initialize() {

  if(reader_version_ == 0) { // 0 -> arrow
    async_reader_ = std::make_unique<arrow_async_reader::ArrowAsyncReader>(env_, target_dir_, output_dtypes_,
                                                               output_shapes_, reader_count_);
  } else {
    async_reader_ = std::make_unique<MultiThreadedAsyncReader>(env_, target_dir_, output_dtypes_,
                                                               output_shapes_, reader_count_);
  }
  return async_reader_->Initialize();
}

Status Reader::Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence) {
  Status s = async_reader_->Read(read_tensors, end_of_sequence);
  if(*end_of_sequence) {
    async_reader_.reset();
  }
  return s;
}


// -----------------------------------------------------------------------------
// MultiThreadedAsyncReader (Base Class for ArrowAsyncReader)
// -----------------------------------------------------------------------------



MultiThreadedAsyncReader::MultiThreadedAsyncReader(Env *env, const std::string &target_dir,
                                                   const DataTypeVector &output_dtypes,
                                                   const std::vector<PartialTensorShape> &output_shapes,
                                                   const int reader_count)
    : env_(env), output_dtypes_(output_dtypes), output_shapes_(output_shapes),
    target_dir_(target_dir), reader_count_(reader_count), tensors_() {
  this->num_readers_done_ = 0;
  // TODO (damien-aymon) add constant for writer version.
}

Status MultiThreadedAsyncReader::Initialize() {

  // Find all the files of this dataset
  std::vector<string> files;
  TF_CHECK_OK(env_->GetMatchingPaths(io::JoinPath(target_dir_, "*\\.easl"), 
      &files));
  file_count_ = files.size();

  { 
    mutex_lock l(mu_);
    for (const auto& f : files) {
      file_names_.push_back(f);
    }
  }

  // Spawn the threadpool, and start reading from the files
  thread_pool_ = absl::make_unique<thread::ThreadPool>(env_, ThreadOptions(),  
      absl::StrCat("reader_thread_pool", reader_count_), reader_count_, false);

  for (int i = 0; i < reader_count_; ++i) {
    thread_pool_->Schedule(
      [this, i] {
        // ReaderThread(env_, i, cache_file_version_, output_dtypes_);
        ReaderThread(env_, i, kWriterVersion, output_dtypes_, output_shapes_);
        }
    );
  }
}

void MultiThreadedAsyncReader::Consume(string* s, bool* end_of_sequence) {
  mutex_lock l(mu_);
  if (file_names_.empty()) {
    *s = ""; 
    *end_of_sequence = true;
  } else {
    *s = file_names_.front();
    file_names_.pop_front();
    *end_of_sequence = false;
  }
}

bool MultiThreadedAsyncReader::ProducerSpaceAvailable() {
  return (tensors_.size() * bytes_per_tensor_) < producer_threshold_;
}

void MultiThreadedAsyncReader::Add(std::vector<Tensor>& tensors) {
  mutex_lock l(mu_add_);
  if(!first_row_info_set_) {
    bytes_per_tensor_ = tensors[0].TotalBytes();  // TODO: this assumes all tensors equal shape --> change!
    VLOG(0) << "EASL - set bytes per tensor: " << bytes_per_tensor_;
    first_row_info_set_ = true;
  }
//  VLOG(0) << "****************** Reader Queue Size: " << tensors_.size() << "  of max:  " << producer_threshold_ / bytes_per_tensor_;
  mu_add_.Await(Condition(this, &MultiThreadedAsyncReader::ProducerSpaceAvailable));
  for (const auto& t : tensors)
    tensors_.push_back(t);

  read_cv_.notify_one();
}

Status MultiThreadedAsyncReader::ReaderThread(Env *env, uint64 writer_id, int64 version,
  DataTypeVector output_types, std::vector<PartialTensorShape> output_shapes) {

  tensorflow::profiler::TraceMe activity(
          "EASLReaderThread", tensorflow::profiler::TraceMeLevel::kVerbose);

  bool end_of_sequence = false; 

  while (!end_of_sequence) {
    std::string file_path;
    Consume(&file_path, &end_of_sequence);

    if (!end_of_sequence) {

      std::unique_ptr<snapshot_util::Reader> reader;

      snapshot_util::Reader::Create(env, file_path, io::compression::kNone,
                                    version, output_types, &reader);


      int64 count = 0;
      bool eof = false;
      while (!eof) {
        std::string t_str = "Reading Tensors:";
        std::vector<Tensor> tensors;
        Status s = reader->ReadTensors(&tensors);
        if (errors::IsOutOfRange(s)) {
          eof = true;  // can't break because of TFRecordReader.
        } else if(s != Status::OK()) {
          return s;
        }

        if(!tensors.empty()) {
          Add(tensors);
        }
      }
    }
  }

  mutex_lock l(mu_add_);
  num_readers_done_++;
  read_cv_.notify_one();

  return Status::OK();
}

Status MultiThreadedAsyncReader::Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence) {
  mutex_lock l(mu_add_);
  *end_of_sequence = false;
  int64 n = output_dtypes_.size();

  while(true){
    if(!tensors_.empty()) {
      while (n > 0) {
        n--;
        read_tensors->push_back(tensors_.front());
        tensors_.pop_front();
      }
      return Status::OK();
    } else {
      if(num_readers_done_ == reader_count_){
        *end_of_sequence = true;
        tensors_.clear();
        return Status::OK();
      }
      // Readers are not done, waiting on data...
      read_cv_.wait(l);
    }
  }
  
  /*
  if (num_readers_done_ == reader_count_) {
    *end_of_sequence = true;
    LOG(INFO) << "(Reader) End of sequence reached, returning last tensors.";
    return Status::OK();
  }
  return Status::OK();*/
}

using namespace std::chrono;
void StatsLogger::WriteInvoked() {
  start_ = high_resolution_clock::now();
  if(num_writes_++) {
    wait_time_sum_ += duration_cast<nanoseconds>(start_ - end_).count();
  }
  start_ = high_resolution_clock::now();
}

void StatsLogger::WriteReturn() {
  end_ = high_resolution_clock::now();
  write_time_sum_ += duration_cast<nanoseconds>(end_ - start_).count();
  end_ = high_resolution_clock::now();
}

void StatsLogger::WriteSleep() {
  sleepStart_ = high_resolution_clock::now();
}

void StatsLogger::WriteAwake() {
  auto now = high_resolution_clock::now();
  auto sleep_time = duration_cast<nanoseconds>(now - sleepStart_).count();
  if(sleep_time > 500) {
    num_sleeps_++;
    sleep_time_sum_ += sleep_time;
  }
}

// printing logging message roughly every second
void StatsLogger::PrintStatsSummary(int writer_id) {
  if(num_writes_ == 0) {
    return;
  }
  uint64_t avg_sleep = 0;
  if(num_sleeps_ > 0) {
    avg_sleep = sleep_time_sum_ / num_sleeps_;
  }
  VLOG(0) << "{avg_write,avg_wait,avg_sleep,num_sleep,num_write} _|LogStat|_ " << write_time_sum_ / num_writes_ << " "
              "" << wait_time_sum_ / (num_writes_ - 1) << " " << avg_sleep << " " << num_sleeps_ << ""
              " " << num_writes_;

  // iterate over threads and print statistics
  ThreadLog& writer_thread = thread_logs_[writer_id];

  VLOG(0) << "[Writer " << writer_id << "] {avg_write,avg_not_write,avg_conversion,num_writes} _|LogStat|_ "
                  "" << writer_thread.write_time_sum / writer_thread.num_writes << " "
                  "" << writer_thread.not_write_time_sum / writer_thread.num_writes << " "
                  "" << writer_thread.conversion_time_sum / writer_thread.num_writes << " "
                  "" << writer_thread.num_writes;

  // reset
  num_sleeps_ = 0;
  num_writes_ = 0;
  sleep_time_sum_ = 0;
  write_time_sum_ = 0;
  wait_time_sum_ = 0;
  last_log_ = high_resolution_clock::now();
}

void StatsLogger::FinishWriteTensors(int id) {
  auto now = high_resolution_clock::now();
  auto since_last = duration_cast<nanoseconds>(now - thread_logs_[id].timestamp).count();
  thread_logs_[id].write_time_sum += since_last;
  thread_logs_[id].num_writes++;
  thread_logs_[id].timestamp = now;
}

void StatsLogger::BeginWriteTensors(int id) {
  auto now = high_resolution_clock::now();
  if(!thread_logs_[id].num_writes) {
    thread_logs_[id].timestamp = now;

    return;
  }
  auto since_last = duration_cast<nanoseconds>(now - thread_logs_[id].timestamp).count();
  thread_logs_[id].not_write_time_sum += since_last;
  thread_logs_[id].timestamp = now;
}

void StatsLogger::FinishConversion(int id) {
  auto now = high_resolution_clock::now();
  auto since_last = duration_cast<nanoseconds>(now - thread_logs_[id].timestamp).count();
  thread_logs_[id].conversion_time_sum += since_last;
}

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow