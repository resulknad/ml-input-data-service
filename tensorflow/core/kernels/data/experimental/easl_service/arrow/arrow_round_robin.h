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

    Status WriterThread(Env* env, const std::string& shard_directory,
                        uint64 checkpoint_id, const std::string& compression,
                        int64 version, DataTypeVector output_types) override;

    void SignalEOF();
private:

  // store tensors until a writerthread "consumes" them
  std::vector<std::vector<Tensor>*>* tensor_batch_;

  // threshold used to control bytes in producer vec. Default or set via version (TODO)
  size_t thresh_ = 1e9;
  size_t available_row_capacity_ = 0;  // how many rows can be inserted without checking bytes_written?
  mutex mu_by_;
  size_t bytes_written_ = 0;  // guraded by mu_by_ --> bytes written out to disk by writers
  size_t bytes_received_ = 0;  // bytes received by iterator. Make sure bytes_received_ - bytes_written < thresh_
  condition_variable capacity_available_;  // if pipeline full, iterator thread waits until capacity available
};

class ArrowWriter {

};

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
