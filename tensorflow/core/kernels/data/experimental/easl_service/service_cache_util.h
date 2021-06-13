#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_


#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/threadpool.h"
#include <chrono>

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

#define STATS_LOG  // comment this if no stats should be printed as log output
#define DEBUGGING // comment this if debugging statements should be removed


// Logging utility class to get info where we spend how much time.
struct ThreadLog {
  bool used = false;
  uint64_t write_time_sum = 0;
  uint64_t not_write_time_sum = 0;
  uint64_t num_writes = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
  uint64_t conversion_time_sum = 0;
};

class StatsLogger {
public:
    void WriteSleep();
    void WriteAwake();
    void FinishConversion(int id);
    void BeginWriteTensors(int id);
    void FinishWriteTensors(int id);
    void WriteInvoked();
    void WriteReturn();   // printing logging message roughly every 2 second
    void PrintStatsSummary(int id);
    private:

    std::chrono::time_point<std::chrono::high_resolution_clock> sleepStart_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_log_ = std::chrono::high_resolution_clock::now();

    ThreadLog thread_logs_[12];  // writers directly access this by writer_id
    uint64_t num_writes_ = 0;
    uint64_t num_sleeps_ = 0;
    uint64_t sleep_time_sum_ = 0;
    uint64_t write_time_sum_ = 0;  // duration of num_writes_ writes
    uint64_t wait_time_sum_ = 0;  // duration of (num_writes_ - 1) waits
    const int log_wait_ = 0; // wait ~1s betw. logs
};

// struct that is extended with data by class implementing BoundedMemoryWriter
struct ElementOrEOF {
    bool eof = false;
    virtual ~ElementOrEOF() = default;  // needs to be virtual s.t. unique_ptr<ElementOrEOF> calls dtor of derived
};

// BoundedMemoryWriter is base class for TFRecordWriter and the Arrow family writers.
// It ensures that bytes_written_ - bytes_received_ < memory_threshold_.
// Further it provides convenient abstraction to implement any new data type with little effort.
class BoundedMemoryWriter {
public:
    BoundedMemoryWriter(int writer_count, uint64 memory_threshold);

  void Initialize(Env* env, int64 file_index,
                  const std::string& shard_directory, uint64 checkpoint_id,
                  const std::string& compression, int64 version,
                  const DataTypeVector& output_types,
                  std::function<void(Status)> done);

    // Provides stats telling how big the in-flow throughput is, as well as
    Status Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_, mu_by_);

    void SignalEOF();

    virtual ~BoundedMemoryWriter() = default;  // this is important to be virtual --> else memory leak

protected:
    // abstract method used to insert data into deque_ or accumulate them internally into batches.
    virtual void InsertData(const std::vector<Tensor>& tensors) = 0;  // Guarded by mu_

    // this method gets invoked the first time a dataset element is passed to the writer.
    // it's used to extract shape and datatype information if needed.
    virtual void FirstRowInfo(const std::vector<Tensor>& tensors) = 0;

    // creates an empty ElementOrEOF with eof set to true.
    virtual std::unique_ptr<ElementOrEOF> CreateEOFToken() = 0;

    // function where the main conversion happens. It must use the "BeforeWrite" and "AfterWrite" functions
    // to support logging.
    virtual void WriterThread(Env *env, const std::string &shard_directory,
                      int writer_id, const std::string& compression, const DataTypeVector& output_types, int64 version) = 0;  // Guarded by mu_by_

    // utility function that can be implemented by class inheriting from this to support stats logging.
    void BeforeWrite(int thread_id) {
      #ifdef STATS_LOG
      logger_->BeginWriteTensors(thread_id);
      #endif
    }

    void FinishedConversion(int thread_id) {
      #ifdef STATS_LOG
      logger_->FinishConversion(thread_id);
      #endif
    }

    // utility function that can be implemented by class inheriting from this to support stats logging.
    void AfterWrite(int thread_id) {
      #ifdef STATS_LOG
      logger_->FinishWriteTensors(thread_id);
      #endif
    }

    // inheriting class must put this at the end of the WriterThread function.
    void WriterReturn(int thread_id) TF_LOCKS_EXCLUDED(mu_){
      mutex l(mu_);
      writers_finished_++;
      #ifdef DEBUGGING
      VLOG(0) << "Thread " << thread_id << " finished writing, returning.";
      #endif
    }

    // transfers ownership of the first unique_ptr to an ElementOrEOF to the passed unique_ptr and pops it from deque
    std::unique_ptr<ElementOrEOF> Consume(int writer_id) TF_LOCKS_EXCLUDED(mu_);

    // mutex has to be available in sub-class
    mutex mu_;  // mutex guarding deque
    mutex mu_by_; // mutex guarding bytes written
    std::deque<std::unique_ptr<ElementOrEOF>> deque_ TF_GUARDED_BY(mu_);
    uint64 bytes_per_row_; // initialized to 0 by constructor. Used to calculate mem-usage.
    uint64 bytes_written_ TF_GUARDED_BY(mu_by_); // initialized to 0 by constructor.

private:
    // private functions
    bool ElementAvailable() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);  // writer_threads consume if element available
    bool ProducerSpaceAvailable() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_by_);  // if no more space, wait() goes to sleep
    bool AllWritersFinished() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);  // only destruct once all threads returned

    // class internal variables:
    bool first_row_info_set_; // initialized to false. Signals that first row has already arrived.
    const int writer_count_; // initialized by constructor.
    const int memory_threshold_; // initialized by constructor.
    uint64 bytes_received_; // initialized to 0 by constructor.
    int writers_finished_ TF_GUARDED_BY(mu_);  // initialized to 0 by constructor.
    // indicates how many rows can be received at least until it has to check the overall memory in pipeline again.
    uint64 available_row_capacity_;  // initialized to 0 by constructor, later when first row received.

    // logging utility.
    #ifdef STATS_LOG
    std::unique_ptr<StatsLogger> logger_;
    #endif

    // This has to be last. During destruction, we need to make sure that the
    // Thread object is destroyed first as its destructor blocks on thread
    // completion. If there are other member variables after this, they may get
    // destroyed first before the thread finishes, potentially causing the
    // thread to access invalid memory.
    std::unique_ptr<thread::ThreadPool> thread_pool_;
};


struct RowOrEOF : public ElementOrEOF {
  std::vector<Tensor> data;
};

class TFRecordWriter : public BoundedMemoryWriter {
public:
    explicit TFRecordWriter(int writer_count, uint64 memory_threshold);

    // method used to insert data into deque_.
    void InsertData(const std::vector<Tensor>& tensors) override;

    // unuesed in TFRecord writer
    void FirstRowInfo(const std::vector<Tensor>& tensors) override;

    // creates an empty ElementOrEOF with eof set to true.
    std::unique_ptr<ElementOrEOF> CreateEOFToken() override;

    void WriterThread(Env *env, const std::string &shard_directory,
                      int writer_id, const std::string& compression, const DataTypeVector& output_types, int64 version) override;

};




// EASL (damien-aymon)
// Top-level writer that handles writes to the service cache.
// In the future, this class should handle more complex logic on how and where
// to distribute individual elements to cache files.
// For now, this class simply wraps a single snapshot_util::async writer using
// a TFRecordWriter.
class Writer {
 public:
  Writer(Env* env,
         const std::string& target_dir,
         const DataTypeVector& output_dtypes,
         const std::vector<PartialTensorShape>& output_shapes,
         int writer_count = 8,
         int writer_version = 0,
         int compression = 0);

  Status Write(const std::vector<Tensor>& tensors);

  Status Close();

  ~Writer();

  Status Initialize();

 private:
  const int writer_version_;
  Env* env_;
  const int writer_count_;
  const int compression_;
  const std::string target_dir_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<BoundedMemoryWriter> async_writer_;


  // Memory threshold of writers (and all async_writers)
  uint64 MEMORY_THRESHOLD_ = 2e9;
};

class MultiThreadedAsyncReader {
 public:
  MultiThreadedAsyncReader(Env *env,
                           const std::string &target_dir,
                           const DataTypeVector &output_dtypes,
                           const std::vector<PartialTensorShape> &output_shapes,
                           int reader_count = 8);

  Status Initialize();

  Status Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence);

  virtual ~MultiThreadedAsyncReader()= default;

 protected:
  mutex mu_;
  mutex mu_add_;
  condition_variable read_cv_ TF_GUARDED_BY(mu_);
  int file_count_;
  const int reader_count_;
  int8 num_readers_done_ TF_GUARDED_BY(mu_add_);

  void Consume(string* s, bool* end_of_sequence) TF_LOCKS_EXCLUDED(mu_);
  void Add(std::vector<Tensor>& tensors)  TF_LOCKS_EXCLUDED(mu_add_);
  virtual Status ReaderThread(Env *env, uint64 writer_id, int64 version,
      DataTypeVector output_types, std::vector<PartialTensorShape> output_shapes);

  const std::string target_dir_;
  int64 cache_file_version_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  Env* env_;


  bool ProducerSpaceAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const uint64 producer_threshold_ = 1e9;  // allow producer queue to hold 1 GB
  bool first_row_info_set_ = false;
  uint64 bytes_per_tensor_ = 0;

    //   std::unique_ptr<snapshot_util::Reader> reader_;
  std::deque<string> file_names_ TF_GUARDED_BY(mu_);
  std::deque<Tensor> tensors_ TF_GUARDED_BY(mu_add_);
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};


class Reader {
public:
    Reader(Env *env,
           const std::string &target_dir,
           const DataTypeVector& output_dtypes,
           const std::vector<PartialTensorShape>& output_shapes,
           const int reader_count = 8,
           const int reader_version = 0);

    Status Initialize();

    Status Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence);

    ~Reader()= default;
private:
    const int reader_version_;
    Env* env_;
    const int reader_count_;
    const std::string target_dir_;
    const DataTypeVector output_dtypes_;
    const std::vector<PartialTensorShape> output_shapes_;
    std::unique_ptr<MultiThreadedAsyncReader> async_reader_;
};


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
