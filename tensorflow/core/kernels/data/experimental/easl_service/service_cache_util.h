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

// MultiThreadedAsyncWriter provides API for asynchronously writing dataset 
// elements (each represented as a vector of tensors) to a file.
//
// The expected use of this API is:
//
// std::unique_ptr<MultiThreadedAsyncWriter> writer = 
// absl_make_unique<MultiThreadedAsyncWriter>(...);
//
// while (data_available()) {
//   std::vector<Tensor> data = read_data()
//   writer->Write(data);
// }
// writer->SignalEOF();
// writer = nullptr;  // This will block until writes are flushed.
class MultiThreadedAsyncWriter {
 public:
  MultiThreadedAsyncWriter(const int writer_count);

  virtual void Initialize(Env* env, int64 file_index,
                  const std::string& shard_directory, uint64 checkpoint_id,
                  const std::string& compression, int64 version,
                  const DataTypeVector& output_types,
                  std::function<void(Status)> done);

  // Writes the given tensors. The method is non-blocking and returns without
  // waiting for the element to be written.
  virtual void Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_);

  // Signals the end of input. The method is non-blocking and returns without
  // waiting for the writer to be closed.
  virtual void SignalEOF() TF_LOCKS_EXCLUDED(mu_);

  virtual ~MultiThreadedAsyncWriter()= default;

  std::unique_ptr<StatsLogger> logger;

protected:
  void Consume(snapshot_util::ElementOrEOF* be) TF_LOCKS_EXCLUDED(mu_);
  bool ElementAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual Status WriterThread(Env* env, const std::string& shard_directory,
                      uint64 checkpoint_id, const std::string& compression,
                      int64 version, DataTypeVector output_types);

  mutex mu_;
  std::deque<snapshot_util::ElementOrEOF> deque_ TF_GUARDED_BY(mu_);

  // look at first row of dataset to infer bytes per row and dataset shape
  virtual bool ProducerSpaceAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const uint64 producer_threshold_ = 1e9;  // allow producer queue to hold 1 GB
  bool first_row_info_set_ = false;
  std::vector<TensorShape> first_row_shape_;
  uint64 bytes_per_row_ = 0;

  // make sure that SignalEOF returns ones all writer threads have finished.
  const int writer_count_;
  int writer_finished_ = 0;  // also guarded by mu_
  bool AllWritersFinished() {  // needed to wait for condition
    VLOG(0) << "Checking AllWritersFinished: " << writer_finished_ << " of " << writer_count_ << " finished.";
    return writer_finished_ == writer_count_;
  }
  condition_variable finish_cv_ TF_GUARDED_BY(mu_);


  // This has to be last. During destruction, we need to make sure that the
  // Thread object is destroyed first as its destructor blocks on thread
  // completion. If there are other member variables after this, they may get
  // destroyed first before the thread finishes, potentially causing the
  // thread to access invalid memory.
  std::unique_ptr<thread::ThreadPool> thread_pool_;
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
         const int writer_count = 8,
         const int writer_version = 0);

  Status Write(const std::vector<Tensor>& tensors);

  Status Close();

  ~Writer();

  Status Initialize();

 private:
  Status WriteMetadataFile(
      Env* env, const std::string& path, const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes);

  const int writer_version_;
  Env* env_;
  const int writer_count_;
  const std::string target_dir_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<MultiThreadedAsyncWriter> async_writer_;
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

  Status ReadAndParseMetadataFile();
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
