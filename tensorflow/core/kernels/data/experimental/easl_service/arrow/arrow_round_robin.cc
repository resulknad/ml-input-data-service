//
// Created by simon on 18.05.21.
//

#include "arrow_round_robin.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"
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
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  metadata_->SetExperimental(true);
  max_batch_size_ = thresh_ / (writer_count + 1);
  std::vector<std::vector<Tensor>> tensor_batch_vec;
  tensor_batch_vec.reserve(max_batch_size_ / sizeof(std::vector<Tensor>) + 1);
  current_batch_ = {tensor_batch_vec, 0, false};
}

void ArrowRoundRobinWriter::PushCurrentBatch() {
  if(current_batch_.byte_count == 0) {
    return;
  }
  mu_.lock();
  deque_.push_back(current_batch_);
  mu_.unlock();
  tensors_available_.notify_one();
  std::vector<std::vector<Tensor>> tensor_batch_vec;
  tensor_batch_vec.reserve(max_batch_size_ / sizeof(std::vector<Tensor>) + 1);
  current_batch_ = {tensor_batch_vec, 0, false};
}

// invariants: incoming tensors have enough "space" in pipeline. write sleeps if we need to stall.
void ArrowRoundRobinWriter::Write(const std::vector<Tensor>& tensors) {
  logger->WriteInvoked();

  // only take this branch once in the beginning
  if(!first_row_info_set_) {
    for(const Tensor& t : tensors) {
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
    VLOG(0) << "_|Iterator|_ First-Row-Received";

  }

  bytes_received_ += bytes_per_row_;
//  std::vector<Tensor> local_tensors = *tensors;
  current_batch_.tensor_batch.push_back(tensors);  // copy of tensors now stored in class -> survive until written
  current_batch_.byte_count += bytes_per_row_;
//  VLOG(0) << "ARR - Write - current_batch size: " << current_batch_.byte_count << "  /  " << max_batch_size_;

  // check if current batch full (--> next row wouldn't fit into current batch),
  // may be negative if many workers and high bytes per row rate.
  if(current_batch_.byte_count + bytes_per_row_ > max_batch_size_) {
    // feed new tensors in pipeline
    VLOG(0) << "_|Iterator|_ Batch-Full-Notifying-Writer";

    PushCurrentBatch();

  }


  // if we still have capacity in the pipeline, simply move on.
  if(available_row_capacity_ > 1) {  // capacity should be at least 1 when returning (for next invoke)
    available_row_capacity_--;
//    VLOG(0) << "ARR - Write - row capacity reduced, next = " << available_row_capacity_;
    logger->WriteReturn();
    return;
  }


  // get amount of bytes written out to files
  size_t bytes_written_local;
  mu_by_.lock_shared();  // lock in read-only mode
  bytes_written_local = bytes_written_;
  mu_by_.unlock_shared();

//  VLOG(0) << "Arr - Write - now row-capacity, checking bytes written to disk: " << bytes_written_local;

  if(bytes_received_ - bytes_written_local > thresh_ - bytes_per_row_) {
    // can't take any more rows --> max capacity reached

    VLOG(0) << "_|Iterator|_ No-Capacity-Sleep";
    PushCurrentBatch();

    logger->WriteSleep();

    //reacquire lock and release it
    mutex_lock l(mu_by_);
    capacity_available_.wait(l);

    logger->WriteAwake();

    VLOG(0) << "_|Iterator|_ Awake-New-Capacity";

    bytes_written_local = bytes_written_;
  }

  // calculate how many new rows fit into the pipeline:
  size_t bytes_in_pipeline = bytes_received_ - bytes_written_local;
  available_row_capacity_ = (thresh_ - bytes_in_pipeline) / bytes_per_row_;

  logger->WriteReturn();
} // mutex_lock's destructor automatically releases the lock

void ArrowRoundRobinWriter::ConsumeTensors(TensorData* dat_out, int writer_id) {
  mutex_lock l(mu_);
  if(deque_.empty()) {
    VLOG(0) << "_|" << writer_id << "|_ No-Data-Sleep";
    tensors_available_.wait(l);
    VLOG(0) << "_|" << writer_id << "|_ Wake-Up-Write-Tensors";

  }
  *dat_out = deque_.front();
  deque_.pop_front();
}


/***********************
 * Arrow Writer *
 ***********************/
Status ArrowRoundRobinWriter::ArrowWrite(const std::string &filename, TensorData &dat) {

  // initializing writer process
  int ncols = tensor_data_len_.size();
  arrow::DataTypeVector arrow_dtypes;
  for (int i = 0; i < first_row_dtype_.size(); i++) {
    std::shared_ptr<arrow::DataType> arrow_dt;
    TF_RETURN_IF_ERROR(ArrowUtil::GetArrowType(first_row_dtype_[i], &arrow_dt));
    arrow_dtypes.push_back(arrow_dt);
  }

//  VLOG(0) << "ARR - ArrowWriter - Converted dtypes to arrow";

  // TODO: this writer currently not supporting strings, goal was to keep as simple as possible
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  std::vector<std::shared_ptr<arrow::StringBuilder>> data_builders;
  for(int i = 0; i < ncols; i++) {
    std::shared_ptr<arrow::StringBuilder> data_builder =
            std::make_shared<arrow::StringBuilder>(arrow::default_memory_pool());
    data_builders.push_back(data_builder);
    std::shared_ptr<arrow::Array> arr_ptr;
    arrays.push_back(arr_ptr);
  }

//  VLOG(0) << "ARR - ArrowWriter - Created data builders";

  // iterate over all columns and build array
  std::vector<std::vector<Tensor>> &data = dat.tensor_batch;

//  VLOG(0) << "ARR - ArrowWriter - created data view";

  std::vector<size_t> data_len = tensor_data_len_;

//  VLOG(0) << "ARR - ArrowWriter - copied tensor_data_len";

  bool partial_batching = false;
  for (int i = 0; i < data.size(); i++) {
    std::vector<Tensor> &row = data[i];

//    VLOG(0) << "ARR - ArrowWriter - builder loop created row view";

    // check for partial batches at the end:
    if(i == data.size() - 1) {
//      VLOG(0) << "ARR - ArrowWriter - processing last row";
      std::vector<size_t> last_row_data_len;
      for(int j = 0; j < ncols; j++) {
        last_row_data_len.push_back(row[j].TotalBytes());
        if(last_row_data_len[j] != data_len[j]) {
//          VLOG(0) << "ARR - ArrowWriter - found partial batch";
          partial_batching = true;
        }
      }
      if(partial_batching) {
        // adjust lengths for conversion of last batch
        data_len = last_row_data_len;

        // store shapes in metadata
        std::vector<TensorShape> shapes;
        for(int j = 0; j < ncols; j++) {
          shapes.push_back(row[j].shape());
        }
        metadata_->AddPartialBatch(filename, shapes);
      }
    }

//    VLOG(0) << "ARR - ArrowWriter - builder loop adding data ...";

    for(int j = 0; j < ncols; j++) {
      const char* buff = row[j].tensor_data().data();
//      VLOG(0) << "ARR - ArrowWriter - builder loop data addr: " << (void *) buff;
      data_builders[j]->Append(buff, data_len[j]);
    }

//    VLOG(0) << "ARR - ArrowWriter - builder loop end";
  }

//  VLOG(0) << "ARR - ArrowWriter - Written data to data builder";

  // finish arrays and construct schema_vector
  for(int i = 0; i < ncols; i++) {
//    VLOG(0) << "ARR - ArrowWriter - Finishing array " << i << "...";
    data_builders[i]->Finish(&arrays[i]);
//    VLOG(0) << "ARR - ArrowWriter - Finish Completed successfully";
    schema_vector.push_back(arrow::field(std::to_string(i), arrays[i]->type()));
  }

//  VLOG(0) << "ARR - ArrowWriter - creating schema and table ...";

      // create schema from fields
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, arrays);

  std::shared_ptr<arrow::io::FileOutputStream> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::FileOutputStream::Open(filename, /*append=*/false));

  CHECK_ARROW(arrow::ipc::feather::WriteTable(*table, file.get(),
          arrow::ipc::feather::WriteProperties::Defaults()));
  CHECK_ARROW(file->Close());

//  VLOG(0) << "ARR - ArrowWriter - Written data to file";

  return Status::OK();
}



Status ArrowRoundRobinWriter::WriterThread(tensorflow::Env *env,
                                           const std::string &shard_directory,
                                           tensorflow::uint64 writer_id,
                                           const std::string &compression,
                                           tensorflow::int64 version,
                                           tensorflow::DataTypeVector output_types) {

  VLOG(0) << "_|" << writer_id << "|_ Thread-Invoked";

  // register writer in metadata, as last writer left has to write out arrowMetadata.
  metadata_->RegisterWorker();

  int split_id = 0;
  TensorData dat;
  ConsumeTensors(&dat, writer_id);
  while(!dat.end_of_sequence) {
    size_t dat_size = dat.byte_count;
    logger->BeginWriteTensors(writer_id);
    Status s = ArrowWrite(GetFileName(shard_directory, writer_id, split_id++), dat);
    logger->FinishWriteTensors(writer_id);
    if(!s.ok()) {
      VLOG(0) << "Writer " << writer_id << "  not ok ... " << s.ToString();
    }
    VLOG(0) << "_|" << writer_id << "|_ Tensors-Written";

    mu_by_.lock();
    bytes_written_ += dat_size;
    mu_by_.unlock();
    capacity_available_.notify_all();
    ConsumeTensors(&dat, writer_id);
  }

  metadata_->WriteMetadataToFile(shard_directory); // de-registers worker, if last write out to disk

  VLOG(0) << "_|" << writer_id << "|_ De-Registered";
  logger->PrintStatsSummary(writer_id);

  mutex_lock l(mu_);
  writer_finished_++;
  return Status::OK();
}

void ArrowRoundRobinWriter::SignalEOF() {
  VLOG(0) << "_|Iterator|_ SignalEOF";
  // wait for all bytes to be written out
  mutex_lock l(mu_);
  if(current_batch_.byte_count > 0) {
    deque_.push_back(current_batch_);
  }
  for(int i = 0; i < writer_count_; i++) {
    std::vector<std::vector<Tensor>> empty;
    deque_.push_back({empty, 0, true});
  }
  tensors_available_.notify_all();
}

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow