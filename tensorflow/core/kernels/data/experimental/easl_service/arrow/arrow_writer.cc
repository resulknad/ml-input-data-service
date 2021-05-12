//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_writer.h"

#include <utility>
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

// ------------------------ arrow writer ----------------------
ArrowWriter::ArrowWriter() = default; // empty constructor, call Create for error handling.

Status ArrowWriter::Create(Env *env, const std::string &filename,
                         const string &compression_type,
                         const DataTypeVector &dtypes,
                         std::shared_ptr<ArrowUtil::ArrowMetadata> metadata) {


  this->metadata_ = metadata;
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
  this->ncols_ = dtypes.size(); // one column in final arrow table per data type (one "dataset row")
  this->current_col_idx_ = 0;
  this->tensor_data_len_initialized_ = false;

  // default CPU_Allocator
  string_allocator_ = cpu_allocator(port::kNUMANoAffinity);

  // shapes of first row of dataset
  metadata_->GetRowShape(&shapes_);

  // Get Arrow Data Types
  for(int i = 0; i < dtypes_.size(); i++) {
    std::shared_ptr<arrow::DataType> arrow_dt;
    TF_RETURN_IF_ERROR(ArrowUtil::GetArrowType(dtypes[i], &arrow_dt));
    this->arrow_dtypes_.push_back(arrow_dt);
  }

  // initialize data columns
  for(int i = 0; i < ncols_; i++) {
    std::vector<const char *> col;
    tensor_data_.push_back(col);
  }

  return Status::OK();
}

/// \brief convert from given compression to arrow compression type supported by the feather writer
arrow::Compression::type ArrowWriter::getArrowCompressionType(){
  if(compression_type_ == "LZ4_FRAME" == 0) {
    return arrow::Compression::LZ4_FRAME;
  } else if(compression_type_ == "ZSTD") {
    return arrow::Compression::ZSTD;
  }

  return arrow::Compression::UNCOMPRESSED;
}

Status ArrowWriter::Close() {

  // check if writer has any data, if not just return.
  if(!tensor_data_len_initialized_) {
    return Status::OK();
  }
  // get converted arrow array for each column:
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  // iterate over all columns and construct corresponding arrow::Array and arrow::field (for schema)
  for(int i = 0; i < ncols_; i++) {
    std::shared_ptr<arrow::Array> arr_ptr;

    if(arrow_dtypes_[i]->Equals(arrow::utf8()) || !metadata_->IsExperimental()) {
      // check for partial batches
      auto last_tensor_dims = !partial_shapes_.empty() ? partial_shapes_[i].dim_sizes() : shapes_[i].dim_sizes();
      CHECK_ARROW(ArrowUtil::GetArrayFromData(
              arrow_dtypes_[i], tensor_data_[i], shapes_[i].dim_sizes(), &arr_ptr, last_tensor_dims));

      if(arrow_dtypes_[i]->Equals(arrow::utf8())){
        // Deallocate String memory
        for (auto &buff : tensor_data_[i]) {
          string_allocator_->DeallocateRaw((void *) buff);
        }
      }
    } else {
      // check for partial batches
      auto last_row_len = !partial_shapes_.empty() ? last_row_len_[i] : tensor_data_len_[i];
      CHECK_ARROW(ArrowUtil::GetArrayFromDataExperimental(
              tensor_data_len_[i], tensor_data_[i], &arr_ptr, last_row_len));
    }

    arrays.push_back(arr_ptr);
    schema_vector.push_back(arrow::field(std::to_string(i), arr_ptr->type()));
  }



  // create schema from fields
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, arrays);



  // write table to file:
  std::shared_ptr<arrow::io::FileOutputStream> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::FileOutputStream::Open(filename_, /*append=*/false));

  struct arrow::ipc::feather::WriteProperties wp = {
      arrow::ipc::feather::kFeatherV2Version,
      // Number of rows per intra-file chunk. Use smaller chunksize when you need faster random row access
      1LL << 8,
      getArrowCompressionType(),
      arrow::util::kUseDefaultCompressionLevel
  };
  CHECK_ARROW(arrow::ipc::feather::WriteTable(*table, file.get(), wp));
  CHECK_ARROW(file->Close());


  tensor_data_.clear();
  tensors_.clear();

  // write metadata for this writer:
  if(!partial_shapes_.empty()) {
    metadata_->AddPartialBatch(filename_, partial_shapes_);
  }

  return Status::OK();
}

// Assumption: tensors contains one row of the table.
Status ArrowWriter::WriteTensors(std::vector<Tensor> &tensors) {

  for(Tensor t : tensors) {
    // need to get size of tensor buffer for conversion.
    if(!tensor_data_len_initialized_) {

      tensor_data_len_.push_back(t.TotalBytes());
      tensor_data_len_initialized_ = tensor_data_len_.size() >= ncols_;
    }

    // check whether shape of current tensor conforms to shape of other tensors in the same column
    if(t.shape() != shapes_[current_col_idx_]) {
      partial_shapes_.push_back(t.shape());
      last_row_len_.push_back(t.TotalBytes());
    }

   if(arrow_dtypes_[current_col_idx_]->Equals(arrow::utf8())) {

	    // get string data for tensor
      auto str_data = reinterpret_cast<const tstring*>(t.data());

      // number of strings in tensor:
      int64_t n_elements = t.NumElements();

     // 8-byte boundary. Memory freed after strings have been written to arrow array.
     const char ** str_refs = (const char **) string_allocator_->AllocateRaw(
             Allocator::kAllocatorAlignment, n_elements * 8);

     // accumulate pointers to strings in a char **
      for(int i = 0; i < n_elements; i++) {
        str_refs[i] = str_data[i].data();
      }
     // accumulate buffers in correct column:
     tensor_data_[current_col_idx_].push_back((const char*) str_refs);
   } else { // if not a string, it is a simple data type -> don't touch data

     // accumulate buffers in correct column:
     tensor_data_[current_col_idx_].push_back(t.tensor_data().data());
   }

    current_col_idx_ = (current_col_idx_ + 1) % ncols_;

    // make ArrowWriter owner of tensors s.t. buffers don't get de-allocated.
    tensors_.push_back(t);
  }
  return Status::OK();
}

} // namespace easl
} // namespace data
} // namespace tensorflow
