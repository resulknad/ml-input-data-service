//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_reader.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

ArrowReader::ArrowReader(){}

Status ArrowReader::Initialize(Env *env, const std::string &filename, const string &compression_type,
                               const DataTypeVector &dtypes, const std::vector<PartialTensorShape> &shapes,
                               ArrowUtil::ArrowMetadata *metadata) {

//  this->shapes_ = std::vector<TensorShape>();

//  // initialize shapes
//  if(!shapes.empty()) {
//    shapes_initialized_ = true;
//    experimental_ = true;
//    for (PartialTensorShape pts : shapes) {
//      TensorShape out;
//      if (pts.AsTensorShape(&out)) {
//        shapes_.push_back(out);
//      } else {
//        return Status(error::FAILED_PRECONDITION, "can't deal with partially filled tensors");
//      }
//    }
//  } TODO

  // read metadata
  this->metadata_ = metadata;
  bool partial_batching;
  TF_RETURN_IF_ERROR(metadata_->IsPartialBatching(&partial_batching));
  if(partial_batching) {
    TF_RETURN_IF_ERROR(metadata_->GetPartialBatches(filename, &partial_shapes));
  }
  TF_RETURN_IF_ERROR(metadata_->GetRowShape(&shapes_));

  // initialize internal data structures
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
  this->current_batch_idx_ = -1; // gets increased upon every invocation of read_tensors
  this->current_row_idx_ = 0;

  VLOG(0) << "ArrowReader - Initialize - Initialized with the following parameters:\n"
                   "Filename: " << filename_ << "\nCompression_Type: " << compression_type_;

  // TODO: maybe use env to open file, here I use the built-in functionality of arrow.
  // open file and read table
  std::shared_ptr<arrow::io::MemoryMappedFile> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::MemoryMappedFile::Open(filename_, arrow::io::FileMode::READ))
  std::shared_ptr<arrow::ipc::feather::Reader> reader;
  ARROW_ASSIGN_CHECKED(reader, arrow::ipc::feather::Reader::Open(file));
  std::shared_ptr<::arrow::Table> table;
  CHECK_ARROW(reader->Read(&table));
  total_rows_ = table->num_rows();
  // read individual record batches and append to class internal datastructure (size of record batches
  // given by configuration of writer)
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


Status ArrowReader::InitShapesAndTypes() {
  VLOG(0) << "ArrowReader - InitShapesAndTypes - Implicitly extracting shapes and dtypes from arrow format...";
  for(int i = 0; i < current_batch_->num_columns(); i++) {
    std::shared_ptr<arrow::Array> arr = current_batch_->column(i);

    // get the TensorShape for the current column entry:
    TensorShape output_shape = TensorShape({});
    DataType output_type;

    TF_RETURN_IF_ERROR(ArrowUtil::AssignSpec(arr, 0, 0, &output_type, &output_shape));  //batch_size = 0

    // add to internal data structures
    this->dtypes_.push_back(output_type);
    this->shapes_.push_back(output_shape);

    VLOG(0) << "ArrowReader - InitShapesAndTypes - \n"
               "DataType: " << DataTypeString(output_type) << "\n"
               "Shape: " << output_shape.DebugString() << "\n"
               "Column: " << i;
  }
  shapes_initialized_ = true;
  return Status::OK();
}


Status ArrowReader::ReadTensors(std::vector<Tensor> *read_tensors) {

  // increments current_batch_idx_ by 1 (initialized to -1). If no more batches,
  // return status with OUT_OF_RANGE error.
  TF_RETURN_IF_ERROR(NextBatch());
  // Invariant: current_batch_ != nullptr

//  if(!shapes_initialized_) {
//    TF_RETURN_IF_ERROR(InitShapesAndTypes());
//  } TODO

  // go over all rows of record batch
  for(int i = 0; i < current_batch_->num_rows(); i++) {
    for(int j = 0; j < current_batch_->num_columns(); j++) {
      std::shared_ptr<arrow::Array> arr = current_batch_->column(j);

      DataType output_type = this->dtypes_[j];

      TensorShape output_shape;
      if(partial_shapes.size() > 0 && current_row_idx_ == total_rows_ - 1) {
        output_shape = this->partial_shapes[j];
      } else {
        output_shape = this->shapes_[j];
      }

      // Allocate a new tensor and assign Arrow data to it
      Tensor tensor(output_type, output_shape); // this constructor will use the default_cpu_allocator.

      // String arrays and normal arrays have different shapes in experimental.
      if(output_type == DataType::DT_STRING || !experimental_) {
        VLOG(0) << "ArrowReader - ReadTensors - Standard Assign Tensor";

        TF_RETURN_IF_ERROR(ArrowUtil::AssignTensor(arr, i, &tensor));
      } else {
        VLOG(0) << "ArrowReader - ReadTensors - Experimental Assign Tensor";

        TF_RETURN_IF_ERROR(ArrowUtil::AssignTensorExperimental(arr, i, &tensor));
      }

      read_tensors->emplace_back(std::move(tensor));
      VLOG(0) << "ArrowReader - ReadTensors - Successfully assigned tensor";
    }
    current_row_idx_++;
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