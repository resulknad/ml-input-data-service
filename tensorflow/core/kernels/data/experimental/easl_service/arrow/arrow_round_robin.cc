//
// Created by simon on 18.05.21.
//

#include "arrow_round_robin.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"


namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_round_robin{

namespace {
  std::string GetFileName(const std::string& shard_directory,
  uint64 file_id, uint64 split_id = 0) {
    return io::JoinPath(shard_directory, strings::Printf("%07llu_%llu.easl",
    static_cast<unsigned long long>(file_id),
    static_cast<unsigned long long>(split_id)));
  }
}



ArrowRoundRobinWriter::ArrowRoundRobinWriter(const int writer_count)
        : MultiThreadedAsyncWriter(writer_count) {

  std::vector<std::vector<Tensor>> tensor_batch_vec;
  current_batch_.tensor_batch = &tensor_batch_vec;
  current_batch_.byte_count = 0;
  current_batch_.end_of_sequence = false;
}


// invariants: incoming tensors have enough "space" in pipeline. write sleeps if we need to stall.
void ArrowRoundRobinWriter::Write(std::vector<Tensor> *tensors) {
  VLOG(0) << "ARR - Write - invoked";
  // only take this branch once in the beginning
  if(!first_row_info_set_) {
    for(const Tensor& t : *tensors) {
      size_t bytes = t.TotalBytes();
      bytes_per_row_ += bytes;
      tensor_data_len_.push_back(bytes);
      first_row_shape_.push_back(t.shape());
      first_row_dtype_.push_back(t.dtype());
    }
    // set arrow metadata
    metadata_->SetRowShape(first_row_shape_);
    metadata_->SetRowDType(first_row_dtype_);

    // can't handle this case: memory issue
    assert(thresh_ > bytes_per_row_);
    assert(bytes_per_row_ > 0);

    available_row_capacity_ = thresh_ / bytes_per_row_;
    first_row_info_set_ = true;
    VLOG(0) << "ARR - Write - read first row info. available row capacity: " << available_row_capacity_;

  }

  bytes_received_ += bytes_per_row_;
  current_batch_.tensor_batch->push_back(std::move(*tensors));  // copy of tensors now stored in class -> survive until written
  current_batch_.byte_count += bytes_per_row_;

  // check if current batch full (--> next row wouldn't fit into current batch)
  if(current_batch_.byte_count > max_batch_size_ - bytes_per_row_) {
    // feed new tensors in pipeline
    VLOG(0) << "ARR - Write - current_batch_ full, adding to producer queue";

    mu_.lock();
    deque_.push_back(current_batch_);
    mu_.unlock();
    VLOG(0) << "ARR - Write - Notifying Writer to consume tensors";
    tensors_available_.notify_one();
  }


  // if we still have capacity in the pipeline, simply move on.
  if(available_row_capacity_ > 1) {  // capacity should be at least 1 when returning (for next invoke)
    available_row_capacity_--;
    VLOG(0) << "ARR - Write - row capacity reduced, next = " << available_row_capacity_;
    return;
  }


  // get amount of bytes written out to files
  size_t bytes_written_local;
  mu_by_.lock_shared();  // lock in read-only mode
  bytes_written_local = bytes_written_;
  mu_by_.unlock_shared();

  VLOG(0) << "Arr - Write - now row-capacity, checking bytes written to disk: " << bytes_written_local;

  if(bytes_received_ - bytes_written_local > thresh_ - bytes_per_row_) {
    // can't take any more rows --> max capacity reached

    VLOG(0) << "Arr - Write - Capacity reached, waiting for writers to finish writing";

    //reacquire lock and release it
    mutex_lock l(mu_by_);
    capacity_available_.wait(l);
    VLOG(0) << "Arr - Write - Waking up, capacity available again...";

    bytes_written_local = bytes_written_;
  }

  // calculate how many new rows fit into the pipeline:
  size_t bytes_in_pipeline = bytes_received_ - bytes_written_local;
  available_row_capacity_ = (thresh_ - bytes_in_pipeline) / bytes_per_row_;
  VLOG(0) << "Arr - Write - New row capacity available: " << available_row_capacity_;

} // mutex_lock's destructor automatically releases the lock

void ArrowRoundRobinWriter::ConsumeTensors(TensorData* dat_out) {
  mutex_lock l(mu_);
  if(deque_.empty()) {
    VLOG(0) << "ARR - ConsumeTensors - No Data available, going to sleep...";
    tensors_available_.wait(l);
    VLOG(0) << "ARR - ConsumeTensors - Fresh data, waking up...";

  }
  *dat_out = deque_.front();
  deque_.pop_front();
}


/***********************
 * Arrow Writer *
 ***********************/
Status ArrowRoundRobinWriter::ArrowWrite(const std::string &filename, TensorData dat) {

  // initializing writer process
  int ncols = tensor_data_len_.size();
  Allocator *string_allocator_ = cpu_allocator(port::kNUMANoAffinity);
  arrow::DataTypeVector arrow_dtypes;
  for (int i = 0; i < first_row_dtype_.size(); i++) {
    std::shared_ptr<arrow::DataType> arrow_dt;
    TF_RETURN_IF_ERROR(ArrowUtil::GetArrowType(first_row_dtype_[i], &arrow_dt));
    arrow_dtypes.push_back(arrow_dt);
  }

  VLOG(0) << "ARR - ArrowWriter - Converted dtypes to arrow";

  // TODO: this writer currently not supporting strings, goal was to keep as simple as possible
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  std::vector<arrow::StringBuilder> data_builders;
  for(int i = 0; i < ncols; i++) {
    arrow::StringBuilder data_builder(arrow::default_memory_pool());
    data_builders.push_back(std::move(data_builder));
    std::shared_ptr<arrow::Array> arr_ptr;
    arrays.push_back(arr_ptr);
  }

  VLOG(0) << "ARR - ArrowWriter - Created data builders";

  // iterate over all columns and build array
  std::vector<std::vector<Tensor>> &data = *dat.tensor_batch;
  std::vector<size_t> data_len = std::move(tensor_data_len_);
  bool partial_batching = false;
  for (int i = 0; i < data.size(); i++) {
    std::vector<Tensor> &row = data[i];

    // check for partial batches at the end:
    if(i == data.size() - 1) {
      VLOG(0) << "ARR - ArrowWriter - processing last row";
      std::vector<size_t> last_row_data_len;
      for(int j = 0; j < ncols; j++) {
        last_row_data_len[j] = row[j].TotalBytes();
        if(last_row_data_len[j] != data_len[j]) {
          VLOG(0) << "ARR - ArrowWriter - found partial batch";
          partial_batching = true;
        }
      }
      if(partial_batching) {
        // adjust lengths for conversion of last batch
        data_len = std::move(last_row_data_len);

        // store shapes in metadata
        std::vector<TensorShape> shapes;
        for(int j = 0; j < ncols; j++) {
          shapes.push_back(row[j].shape());
        }
        metadata_->AddPartialBatch(filename, shapes);
      }
    }

    for(int j = 0; j < row.size(); j++) {
      const char* buff = row[j].tensor_data().data();
      data_builders[j].Append(buff, data_len[j]);
    }
  }


  // finish arrays and construct schema_vector
  for(int i = 0; i < ncols; i++) {
    data_builders[i].Finish(&arrays[i]);
    schema_vector.push_back(arrow::field(std::to_string(i), arrays[i]->type()));
  }

      // create schema from fields
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, arrays);

  std::shared_ptr<arrow::io::FileOutputStream> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::FileOutputStream::Open(filename, /*append=*/false));

  CHECK_ARROW(arrow::ipc::feather::WriteTable(*table, file.get(),
          arrow::ipc::feather::WriteProperties::Defaults()));
  CHECK_ARROW(file->Close());

  return Status::OK();
}



Status ArrowRoundRobinWriter::WriterThread(tensorflow::Env *env,
                                           const std::string &shard_directory,
                                           tensorflow::uint64 writer_id,
                                           const std::string &compression,
                                           tensorflow::int64 version,
                                           tensorflow::DataTypeVector output_types) {

  // register writer in metadata, as last writer left has to write out arrowMetadata.
  metadata_->RegisterWorker();

  int split_id = 0;
  TensorData dat;
  ConsumeTensors(&dat);
  while(!dat.end_of_sequence) {
    VLOG(0) << "ARR - WriterThread " << writer_id << " - Consumed fresh data, writing...";
    size_t dat_size = dat.byte_count;
    Status s = ArrowWrite(GetFileName(shard_directory, writer_id, split_id++), dat);
    if(!s.ok()) {
      VLOG(0) << "Writer " << writer_id << "  not ok ... " << s.ToString();
    }
    VLOG(0) << "ARR - WriterThread " << writer_id << " - Successfully written tensors, notifying capacity available";
    mu_by_.lock();
    bytes_written_ += dat_size;
    mu_by_.unlock();
    capacity_available_.notify_all();
    ConsumeTensors(&dat);
  }

  metadata_->WriteMetadataToFile(shard_directory);  // de-registers worker, if last write out to disk
}

void ArrowRoundRobinWriter::SignalEOF() {
  // wait for all bytes to be written out
  mutex_lock l(mu_);
  if(current_batch_.byte_count > 0) {
    deque_.push_back(current_batch_);
  }
  for(int i = 0; i < writer_count_; i++) {
    deque_.push_back({nullptr, 0, true});
  }
  tensors_available_.notify_all();
}

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow