#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_


#include <queue>
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"

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
  ~MultiThreadedAsyncWriter();

  void Initialize(Env* env, int64 file_index,
                  const std::string& shard_directory, uint64 checkpoint_id,
                  const std::string& compression, int64 version,
                  const DataTypeVector& output_types,
                  std::function<void(Status)> done,
                  int64_t task_id);

  // Writes the given tensors. The method is non-blocking and returns without
  // waiting for the element to be written.
  virtual void Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_);

  // Signals the end of input. The method is non-blocking and returns without
  // waiting for the writer to be closed.
  void SignalEOF() TF_LOCKS_EXCLUDED(mu_);

 private:
  std::string GeneratePrefixHash();

 protected:
  void Consume(snapshot_util::ElementOrEOF* be) TF_LOCKS_EXCLUDED(mu_);
  bool ElementAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual Status WriterThread(Env* env, const std::string& shard_directory,
      int64_t task_id,
                      uint64 checkpoint_id, const std::string& compression,
                      int64 version, DataTypeVector output_types);

  mutex mu_;
  std::deque<snapshot_util::ElementOrEOF> deque_ TF_GUARDED_BY(mu_);

  // look at first row of dataset to infer bytes per row and dataset shape
  virtual bool ProducerSpaceAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const uint64 producer_threshold_ = 8 * 1e9;  // allow producer queue to hold 1 GB
  bool first_row_info_set_ = false;
  uint64 queue_size_bytes_ = 0;
  std::vector<TensorShape> first_row_shape_;
  uint64 bytes_per_row_ = 0;

  // This has to be last. During destruction, we need to make sure that the
  // Thread object is destroyed first as its destructor blocks on thread
  // completion. If there are other member variables after this, they may get
  // destroyed first before the thread finishes, potentially causing the
  // thread to access invalid memory.
  const int writer_count_;
  // We define a prefix to ensure no name collisions occur
  const std::string prefix_hash_;
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
         const int64_t task_id,
         const std::string& target_dir,
         const DataTypeVector& output_dtypes,
         const std::vector<PartialTensorShape>& output_shapes,
         const int writer_count = 8,
         const int writer_version = 2);

  Status Write(const std::vector<Tensor>& tensors);

  Status Close();

  ~Writer();

  Status Initialize();
  bool Initialized() { return initialized_; }

 private:
  Status WriteMetadataFile(
      Env* env, const std::string& path, const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes);

  const int writer_version_;
  const int64_t task_id_;
  bool initialized_ = false;

  Env* env_;
  const int writer_count_;
  const std::string target_dir_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<MultiThreadedAsyncWriter> async_writer_;
};

class MultiThreadedAsyncReader {
 public:
  /*MultiThreadedAsyncReader(Env *env,
                           const std::string &target_dir,
                           const DataTypeVector &output_dtypes,
                           const std::vector<PartialTensorShape> &output_shapes,
                           int reader_count = 8);*/

  MultiThreadedAsyncReader(Env *env,
                           std::shared_ptr<SplitProvider> split_provider,
                           const std::string &target_dir,
                           const DataTypeVector &output_dtypes,
                           const std::vector<PartialTensorShape> &output_shapes,
                           int reader_count = 8,
                           bool deterministic = true,
                           int64_t element_index = 0) : env_(env), split_provider_(split_provider), output_dtypes_(output_dtypes), 
    output_shapes_(output_shapes), target_dir_(target_dir), 
    reader_count_(reader_count), num_readers_done_(0), end_of_sequence_(false),
    deterministic_(deterministic), element_index_(element_index), running_readers_(0), blocked_readers_(0) { }


  Status Initialize();

  Status Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence);

  void Close();

  ~MultiThreadedAsyncReader();

 protected:
  mutex mu_;
  mutex mu_add_;
  condition_variable mu_add_cv_;
  bool deterministic_;
  int64_t element_index_;
  std::atomic_long blocked_readers_;
  std::priority_queue<std::pair<int64_t, string>, std::vector<std::pair<int64_t,string>>, std::greater<std::pair<int64_t,string>>> element_index_avail_ {};
  std::atomic_long running_readers_;
  condition_variable read_cv_ TF_GUARDED_BY(mu_);
  int file_count_;
  int reader_count_;
  int num_readers_done_ TF_GUARDED_BY(mu_add_);
  bool cancelled_ TF_GUARDED_BY(mu_add_) = false;

  Status ReadAndParseMetadataFile();
  void Consume(string* s, bool* end_of_sequence) TF_LOCKS_EXCLUDED(mu_);
  void Add(std::vector<Tensor>& tensors, std::pair<int64_t, string> element_identifier)  TF_LOCKS_EXCLUDED(mu_add_);
  void ReaderDone();
  bool ElementAvailable();
  virtual Status ReaderThread(Env *env, uint64 writer_id, int64 version,
      DataTypeVector output_types, std::vector<PartialTensorShape> output_shapes);

  const std::string target_dir_;
  int64 cache_file_version_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  Env* env_;


  bool ProducerSpaceAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_add_);
  const uint64 producer_threshold_ = 8 * 1e9;  // allow producer queue to hold 1 GB
  bool first_row_info_set_ = false;
  uint64 queue_size_bytes_ = 0;
  uint64 bytes_per_element_ = 0;

    //   std::unique_ptr<snapshot_util::Reader> reader_;
  std::deque<string> file_names_ TF_GUARDED_BY(mu_);
  std::deque<snapshot_util::ElementOrEOF> deque_ TF_GUARDED_BY(mu_add_);
  bool end_of_sequence_ TF_GUARDED_BY(mu_add_);
  std::shared_ptr<SplitProvider> split_provider_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};


class Reader {
public:
    Reader(Env *env,
           std::shared_ptr<SplitProvider> split_provider,
           const std::string &target_dir,
           const DataTypeVector& output_dtypes,
           const std::vector<PartialTensorShape>& output_shapes,
           const int reader_count = 8,
           const int reader_version = 2,
           const bool deterministic = true,
           const int64_t element_index = 0)
  : target_dir_(target_dir), split_provider_(split_provider), env_(env), 
    output_dtypes_(output_dtypes), reader_count_(reader_count), 
    reader_version_(reader_version),
    deterministic_(deterministic), element_index_(element_index_){};

    Status Initialize();

    Status Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence);

    void Close();

private:
    const int reader_version_;
    const bool deterministic_;
    const int64_t element_index_;
    Env* env_;
    const int reader_count_;
    const std::string target_dir_;
    const DataTypeVector output_dtypes_;
    std::shared_ptr<SplitProvider> split_provider_;
    const std::vector<PartialTensorShape> output_shapes_;
    std::unique_ptr<MultiThreadedAsyncReader> async_reader_;
};

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
