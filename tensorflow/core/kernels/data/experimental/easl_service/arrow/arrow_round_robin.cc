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
    std::string GetFileName(const std::string& shard_directory, uint64 file_id) {
      return io::JoinPath(shard_directory, strings::Printf("%08llu.arrow.easl",
                 static_cast<unsigned long long>(file_id)));
    }
}

ArrowRoundRobinWriter::ArrowRoundRobinWriter(const int writer_count, const uint64 memory_threshold)
        : BoundedMemoryWriter(writer_count, memory_threshold) {
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  metadata_->SetExperimental(true);
  max_batch_size_ = memory_threshold / (writer_count + 1);
  std::vector<std::vector<Tensor>> tensor_batch_vec;
  tensor_batch_vec.reserve(max_batch_size_ / sizeof(std::vector<Tensor>) + 1);
  current_batch_ = absl::make_unique<BatchOrEOF>();
  current_batch_->eof = false;
  current_batch_->tensor_batch = tensor_batch_vec;
  current_batch_->byte_count = 0;
}

void ArrowRoundRobinWriter::InsertData(const std::vector<Tensor>& tensors) {
  mutex_lock l(mu_);

  current_batch_->tensor_batch.push_back(tensors); // copying tensors here
  current_batch_->byte_count += bytes_per_row_;

  if(current_batch_->byte_count + bytes_per_row_ > max_batch_size_) {
    deque_.push_back(std::move(current_batch_));
    current_batch_ = absl::make_unique<BatchOrEOF>();
    current_batch_->eof = false;
    current_batch_->tensor_batch = std::vector<std::vector<Tensor>> ();
    current_batch_->byte_count = 0;
  }

}

void ArrowRoundRobinWriter::FirstRowInfo(const std::vector<Tensor> &tensors) {
  for(const Tensor& t : tensors) {
    size_t bytes = t.TotalBytes();
    tensor_data_len_.push_back(bytes);
    first_row_shape_.push_back(t.shape());
    first_row_dtype_.push_back(t.dtype());
  }
  // set arrow metadata
  metadata_->SetRowShape(first_row_shape_);
  metadata_->SetRowDType(first_row_dtype_);
}

std::unique_ptr<ElementOrEOF> ArrowRoundRobinWriter::CreateEOFToken() {
  std::unique_ptr<BatchOrEOF> r_eof = absl::make_unique<BatchOrEOF>();
  r_eof->eof = true;
  return std::move(r_eof);
}

/***********************
 * Arrow Writer *
 ***********************/


std::shared_ptr<arrow::RecordBatch> ArrowRoundRobinWriter::RecordBatchFromTensorData(const std::string& filename, BatchOrEOF &dat) {

  // initializing writer process
  int ncols = tensor_data_len_.size();
  arrow::DataTypeVector arrow_dtypes;
  for (auto & dtype : first_row_dtype_) {
    std::shared_ptr<arrow::DataType> arrow_dt;
    ArrowUtil::GetArrowType(dtype, &arrow_dt);
    arrow_dtypes.push_back(arrow_dt);
  }

  // TODO: this writer is currently not supporting strings, goal was to keep as simple as possible
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

  // iterate over all columns and build array
  std::vector<std::vector<Tensor>> &data = dat.tensor_batch;

  std::vector<size_t> data_len = tensor_data_len_;

  bool partial_batching = false;
  for (int i = 0; i < data.size(); i++) {
    std::vector<Tensor> &row = data[i];

    // check for partial batches at the end:
    if(i == data.size() - 1) {
      std::vector<size_t> last_row_data_len;
      for(int j = 0; j < ncols; j++) {
        last_row_data_len.push_back(row[j].TotalBytes());
        if(last_row_data_len[j] != data_len[j]) {
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


    for(int j = 0; j < ncols; j++) {
      const char* buff = row[j].tensor_data().data();
      data_builders[j]->Append(buff, data_len[j]);
    }
  }


  // finish arrays and construct schema_vector
  for(int i = 0; i < ncols; i++) {
    data_builders[i]->Finish(&arrays[i]);
    schema_vector.push_back(arrow::field(std::to_string(i), arrays[i]->type()));
  }

  // create schema from fields
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  schema_ = schema;
  if(rbw_ == nullptr) {
    rbw_ = arrow::ipc::MakeFileWriter(file_, schema_, wo_).ValueOrDie();
  }
  return std::move(arrow::RecordBatch::Make(schema, arrays[0]->length(), arrays));
}

void ArrowRoundRobinWriter::WriterThread(Env *env, const std::string &shard_directory, int writer_id,
                                         const std::string& compression, const DataTypeVector &output_types, int64 version) {

  metadata_->RegisterWorker();
  const string filename = GetFileName(shard_directory, writer_id);
  std::shared_ptr<arrow::RecordBatch> rb;
  file_ = arrow::io::FileOutputStream::Open(filename, /*append=*/false).ValueOrDie();

  while (true) {
    // parent_be now has ownership over the pointer. When out of scope destructed
    std::unique_ptr<ElementOrEOF> parent_be = Consume(writer_id);
    auto* r_be = dynamic_cast<BatchOrEOF*>(parent_be.get());
    size_t dat_size = r_be->byte_count;

    if (r_be->eof) {
      rbw_->Close();
      file_->Close();
      break;
    }

    BeforeWrite(writer_id);
    rb = RecordBatchFromTensorData(filename, *r_be);
    FinishedConversion(writer_id);
    rbw_->WriteRecordBatch(*rb);
    AfterWrite(writer_id);

    mu_by_.lock();
    bytes_written_ += dat_size;
    mu_by_.unlock();
  }
  metadata_->WriteMetadataToFile(shard_directory);
  WriterReturn(writer_id);
}

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow