//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_reader.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

void ArrowReader::PrintTestLog() {
  VLOG(0) << "ARROW - TestLog\nArrow Version: " << arrow::GetBuildInfo().version_string;
}

ArrowReader::ArrowReader(Env *env, const std::string &filename,
                     const string &compression_type, const DataTypeVector &dtypes)
     : env_(env), filename_(filename), compression_type_(compression_type), dtypes_(dtypes){

  // initialize internal data structures
  this->current_batch_idx_ = -1; // gets increased upon every invocation of read_tensors
  this->current_row_idx_ = 0;
}

Status ArrowReader::Initialize() {
  VLOG(0) << "ArrowReader - Initialize - Initialized with the following parameters:\n"
             "Filename: " << filename_ << "\nCompression_Type: " << compression_type_ << "\n"
             "DataTypes: " << DataTypeVectorString(dtypes_);
  std::shared_ptr<arrow::io::MemoryMappedFile> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::MemoryMappedFile::Open(filename_, arrow::io::FileMode::READ))
  std::shared_ptr<arrow::ipc::feather::Reader> reader;
  ARROW_ASSIGN_CHECKED(reader, arrow::ipc::feather::Reader::Open(file));
  std::shared_ptr<::arrow::Table> table;
  CHECK_ARROW(reader->Read(&table));

  arrow::TableBatchReader tr(*table.get());
  std::shared_ptr<arrow::RecordBatch> batch;
  CHECK_ARROW(tr.ReadNext(&batch));
  while(batch != nullptr) {
    record_batches_.push_back(batch);
    CHECK_ARROW(tr.ReadNext(&batch));
  }

  VLOG(0) << "ArrowReader - Initialize - finished reading table into recordbatches.\n"
             "Num record batches: " << record_batches_.size();
  return Status::OK();
}

Status ArrowReader::ReadTensors(std::vector<Tensor> *read_tensors) {

  TF_RETURN_IF_ERROR(NextBatch());
  // Invariant: current_batch_ != nullptr
  VLOG(0) << "ArrowReader - ReadTensors - at this point current_batch should never be null. -> "
             "" << (current_batch_ != nullptr);


  // logging information of record batches:
  VLOG(0) << "ArrowReader - ReadTensors - info of current_batch:\n"
              "Schema : " << current_batch_->schema()->ToString() << "\n"
              "NumColumns : " << current_batch_->num_columns() << "\n"
              "NumRows : " << current_batch_->num_rows() << "\n"
              "RecordBatch : " << current_batch_->ToString() << "\n";

  // go over all rows of record batch
  for(int i = 0; i < current_batch_->num_rows(); i++) {
    for(int j = 0; j < current_batch_->num_columns(); j++) {
      std::shared_ptr<arrow::Array> arr = current_batch_->column(j);
      VLOG(0) << "ArrowReader - ReadTensors - Reading entry (" << i << ", " << j << ") of current_batch_:\n"
                       "Array length (num elements): " << arr->length() << "\n"
                       "Array Type: " << arr->type()->ToString() << "\n"
                       "Array Contents: " << arr->ToString();
      DataType output_type = this->dtypes_[j]; // TODO: probably need to implicitly convert type --> len(dtypes_) != num_columns()

      // get the TensorShape for the column entry:
      TensorShape output_shape = TensorShape({});
      TF_RETURN_IF_ERROR(ArrowUtil::AssignShape(arr, i, 0, &output_shape));  //batch_size = 0

      // Allocate a new tensor and assign Arrow data to it
      Tensor tensor(output_type, output_shape); // TODO use allocator as constructor argument
      TF_RETURN_IF_ERROR(ArrowUtil::AssignTensor(arr, i, &tensor));
      read_tensors->emplace_back(std::move(tensor));
      VLOG(0) << "ArrowReader - ReadTensors - Successfully Read Tensor: " << tensor.DebugString(tensor.NumElements());
    }
  }

  return Status::OK();
}

Status ArrowReader::NextBatch() {
  if (++current_batch_idx_ < record_batches_.size()) {
    VLOG(0) << "ArrowReader - NextBatch - getting next batch at idx=" << current_batch_idx_;
    current_batch_ = record_batches_[current_batch_idx_];
  } else  {
    VLOG(0) << "ArrowReader - NextBatch - finished reading all record batches";
    return Status(error::OUT_OF_RANGE, "finished reading all record batches");
  }
  return Status::OK();
}

} // namespace easl
} // namespace data
} // namespace tensorflow