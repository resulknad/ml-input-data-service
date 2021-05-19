//
// Created by simon on 18.05.21.
//

#include "arrow_round_robin.h"
#include <unistd.h>
#include <pthread.h>

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_round_robin{


ArrowRoundRobinWriter::ArrowRoundRobinWriter(const int writer_count)
        : MultiThreadedAsyncWriter(writer_count) {

  std::vector<std::vector<Tensor>*> tensor_batch_vec;
  tensor_batch_ = &tensor_batch_vec;
}

void ArrowRoundRobinWriter::Write(
        std::vector<Tensor> *tensors) {

  // feed new tensors in pipeline
  mu_.lock();
  tensor_batch_->push_back(tensors);
  mu_.unlock();

  // only take this branch once in the beginning
  if(!first_row_info_set_) {
    for(Tensor t : *tensors) {
      bytes_per_row_ += t.TotalBytes();
    }

    // can't handle this case: memory issue
    assert(thresh_ > bytes_per_row_);
    assert(bytes_per_row_ > 0);

    available_row_capacity_ = thresh_ / bytes_per_row_;
    first_row_info_set_ = true;
  }
  bytes_received_ += bytes_per_row_;

  // if we still have capacity in the pipeline, simply move on.
  if(available_row_capacity_ > 1) {  // capacity should be at least 1 when returning (for next invoke)
    available_row_capacity_--;
    return;
  }

  // get amount of bytes written out to files
  size_t bytes_written_local;
  mu_by_.lock_shared();  // lock in read-only mode
  bytes_written_local = bytes_written_;
  mu_by_.unlock_shared();

  if(bytes_received_ - bytes_written_local > thresh_ - bytes_per_row_) {
    // can't take any more rows --> max capacity reached

    //reacquire lock and release it
    mutex_lock l(mu_by_);
    capacity_available_.wait(l);
    bytes_written_local = bytes_written_;
  }

  // calculate how many new rows fit into the pipeline:
  size_t bytes_in_pipeline = bytes_received_ - bytes_written_local;
  available_row_capacity_ = (thresh_ - bytes_in_pipeline) / bytes_per_row_;


} // mutex_lock's destructor automatically releases the lock

Status ArrowRoundRobinWriter::WriterThread(tensorflow::Env *env,
                                           const std::string &shard_directory,
                                           tensorflow::uint64 checkpoint_id,
                                           const std::string &compression,
                                           tensorflow::int64 version,
                                           tensorflow::DataTypeVector output_types) {

  useconds_t MAX_DELAY = 1000000;
  useconds_t delay = 100000; // Delay in usec

  // retry
  while ( 1 > 2 )
  {
    usleep(delay);
    if (delay < MAX_DELAY)
    {
      delay *= 2;
    } else {
      delay = MAX_DELAY;
    }
  }
  return Status::OK();
}

void ArrowRoundRobinWriter::SignalEOF() {
}

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow