#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_put_op.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

Writer::Writer(const std::string& target_dir, Env* env)
    : target_dir_(target_dir) {
  // TODO (damien-aymon) add constant for writer version.
  async_writer_ = std::make_unique<snapshot_util::AsyncWriter>()
    env, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
    io::compression::kSnappy, /*version*/ 2, dataset()->output_dtypes(),
    /*done*/ [this](Status s){
                      // TODO (damien-aymon) check and propagate errors here!
                      //if (!s.ok()) {
              //LOG(ERROR) << "AsyncWriter in snapshot writer failed: " << s;
              //mutex_lock l(writer_status_mu_);
              //writer_status_ = s;
                    }
  );
}

virtual Status Writer::Write(const std::vector<Tensor>& tensors){
  writer_->Write(tensors);
  // TODO (damien-aymon) check for errors in the async writer
  return Status::OK();
}

virtual Writer::~Writer() {
  // Will call the destructor and block until done writing.
  writer_->SignalEOF();
  writer_.reset();
}




} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow