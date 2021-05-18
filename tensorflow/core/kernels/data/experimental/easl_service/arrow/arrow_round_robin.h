//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
#include "arrow_util.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_round_robin {



class ArrowRoundRobinWriter : public MultiThreadedAsyncWriter {
public:
    explicit ArrowRoundRobinWriter(const int writer_count);

    void Write(std::vector<Tensor>* tensors) override;

    void Initialize(Env* env, int64 file_index,
                    const std::string& shard_directory, uint64 checkpoint_id,
                    const std::string& compression, int64 version,
                    const DataTypeVector& output_types,
                    std::function<void(Status)> done) override;

    Status WriterThread(Env* env, const std::string& shard_directory,
                        uint64 checkpoint_id, const std::string& compression,
                        int64 version, DataTypeVector output_types) override;

    void SignalEOF();
private:

  // store tensors until a writerthread "consumes" them
  std::vector<std::vector<Tensor>*>* tensor_batch_;

  // threshold used to control bytes in producer vec. Default or set via version (TODO)
  size_t thresh_ = 1e9;
  mutex mu_by_;
  size_t tensor_bytes_in_mem_ = 0;  // guarded by mu_by_, gets reduced once writer releases tensors.
};

class ArrowWriter {

};

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
