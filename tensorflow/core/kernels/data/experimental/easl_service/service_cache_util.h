#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_


#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
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
  explicit MultiThreadedAsyncWriter(Env* env, int64 file_index,
                       const std::string& shard_directory, uint64 checkpoint_id,
                       const std::string& compression, int64 version,
                       const DataTypeVector& output_types,
                       std::function<void(Status)> done,
                       const int writer_count);

  // Writes the given tensors. The method is non-blocking and returns without
  // waiting for the element to be written.
  void Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_);

  // Signals the end of input. The method is non-blocking and returns without
  // waiting for the writer to be closed.
  void SignalEOF() TF_LOCKS_EXCLUDED(mu_);

 private:
  void Consume(snapshot_util::ElementOrEOF* be) TF_LOCKS_EXCLUDED(mu_);
  bool ElementAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status WriterThread(Env* env, const std::string& shard_directory,
                      uint64 checkpoint_id, const std::string& compression,
                      int64 version, DataTypeVector output_types);

  mutex mu_;
  std::deque<snapshot_util::ElementOrEOF> deque_ TF_GUARDED_BY(mu_);

  // This has to be last. During destruction, we need to make sure that the
  // Thread object is destroyed first as its destructor blocks on thread
  // completion. If there are other member variables after this, they may get
  // destroyed first before the thread finishes, potentially causing the
  // thread to access invalid memory.
  const int writer_count_;
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
         const int writer_count = 2);

  Status Write(const std::vector<Tensor>& tensors);

  Status Close();

  ~Writer();

  Status Initialize();

 private:
  Status WriteMetadataFile(
      Env* env, const std::string& path, const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes);

  Env* env_;
  const int writer_count_;
  const std::string target_dir_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  std::unique_ptr<MultiThreadedAsyncWriter> async_writer_;
};

class Reader {
 public:
  Reader(Env *env,
         const std::string &target_dir,
         const DataTypeVector& output_dtypes);

  Status Initialize();

  Status Read(std::vector<Tensor>* &read_tensors, bool* end_of_sequence);

  ~Reader();

 private:
  Status ReadAndParseMetadataFile();

  const std::string target_dir_;
  int64 cache_file_version_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  Env* env_;

  std::unique_ptr<snapshot_util::Reader> reader_;
};


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_EASL_SERVICE_SERVICE_CACHE_UTIL_
