#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

Writer::Writer(const std::string& target_dir, Env* env) : target_dir_(target_dir) {
  DataTypeVector empty_dtv; // will not be used anyway
  // TODO (damien-aymon) add constant for writer version.
  async_writer_ = std::make_unique<snapshot_util::AsyncWriter>(
    env, /*file_index*/ 0, target_dir_, /*checkpoint_id*/ 0,
    io::compression::kSnappy, /*version*/ 2, empty_dtv,
    /*done*/ [this](Status s){
                      // TODO (damien-aymon) check and propagate errors here!
                      //if (!s.ok()) {
              //LOG(ERROR) << "AsyncWriter in snapshot writer failed: " << s;
              //mutex_lock l(writer_status_mu_);
              //writer_status_ = s;
              return;
                    }
  );
}

Status Writer::Write(const std::vector<Tensor>& tensors){
  async_writer_->Write(tensors);
  // TODO (damien-aymon) check for errors in the async writer
  return Status::OK();
}

Writer::~Writer() {
  // Will call the destructor and block until done writing.
  async_writer_->SignalEOF();
  async_writer_.reset();
}

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

Reader::Reader(const std::string &target_dir, const DataTypeVector& dtypes, Env *env)
    : target_dir_(target_dir), dtypes_(dtypes), env_(env) {
  // TODO (damien-aymon) add constant for writer version.

}

Status Reader::Initialize(){
  std::string filename = io::JoinPath(target_dir_,
      strings::Printf("%08llu.snapshot",
          static_cast<unsigned long long>(0)));

  //  TODO (damien-aymon) constant for version number.
  return snapshot_util::Reader::Create(
      env_, filename,io::compression::kSnappy,
      /*version*/ 2, dtypes_, &reader_);
}

Status Reader::Read(std::vector<Tensor>* &read_tensors) {
  return reader_->ReadTensors(read_tensors);
}

Reader::~Reader(){}


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow