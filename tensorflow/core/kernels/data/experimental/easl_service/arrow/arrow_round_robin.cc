//
// Created by simon on 18.05.21.
//

#include "arrow_round_robin.h"
#include <pthread.h>

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_round_robin{


ArrowRoundRobinWriter::ArrowRoundRobinWriter(const int writer_count)
        : MultiThreadedAsyncWriter(writer_count) {

}

void ArrowRoundRobinWriter::Write(
        std::vector<Tensor> *tensors) {

  if(!first_row_info_set_) {
    for(Tensor t : *tensors) {
      bytes_per_row_ += t.TotalBytes();
    }
    first_row_info_set_ = true;
  }

  mu_.lock();
  tensor_batch_->push_back(tensors);
  mu_.lock();



}

void ArrowRoundRobinWriter::Initialize(tensorflow::Env *env,
                                       tensorflow::int64 file_index,
                                       const std::string &shard_directory,
                                       tensorflow::uint64 checkpoint_id,
                                       const std::string &compression,
                                       tensorflow::int64 version,
                                       const tensorflow::DataTypeVector &output_types,
                                       std::function<void(
                                               Status)> done) {
}

Status ArrowRoundRobinWriter::WriterThread(tensorflow::Env *env,
                                           const std::string &shard_directory,
                                           tensorflow::uint64 checkpoint_id,
                                           const std::string &compression,
                                           tensorflow::int64 version,
                                           tensorflow::DataTypeVector output_types) {

  return Status::OK();
}

void ArrowRoundRobinWriter::SignalEOF() {
}

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow