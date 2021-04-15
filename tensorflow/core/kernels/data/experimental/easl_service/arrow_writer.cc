//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_writer.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

// ------------------------ arrow writer ----------------------
ArrowWriter::ArrowWriter() {} // empty constructor, call Create for error handling.

Status ArrowWriter::Create(Env *env, const std::string &filename,
                         const string &compression_type,
                         const DataTypeVector &dtypes) {
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
  this->ncols_ = dtypes.size(); // one column in final arrow table per data type (one "dataset row")
  this->current_col_idx_ = 0;
  this->dims_initialized_ = false;

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
}

/// \brief convert from given compression to arrow compression type supported by the feather writer
arrow::Compression::type ArrowWriter::getArrowCompressionType(){
  if(compression_type_.compare("LZ4_FRAME") == 0) {
    return arrow::Compression::LZ4_FRAME;
  } else if(compression_type_.compare("ZSTD") == 0) {
    return arrow::Compression::ZSTD;
  }

  return arrow::Compression::UNCOMPRESSED;
}

Status ArrowWriter::Close() {

  VLOG(0) << "ArrowWriter - Close - invoked";

  // get converted arrow array for each column:
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  // iterate over all columns and construct corresponding arrow::Array and arrow::field (for schema)
  for(int i = 0; i < ncols_; i++) {
    std::shared_ptr<arrow::Array> arr_ptr;
    ArrowUtil::GetArrayFromData(arrow_dtypes_[i], tensor_data_[i], col_dims_[i], &arr_ptr); // TODO: propagate error
    VLOG(0) << "ArrowWriter - Close - conversion completed for column: " << i << ""
                     "\nArray:\n" << arr_ptr->ToString() << ""
                     "\nDType: " << arr_ptr->type()->ToString();


    arrays.push_back(arr_ptr);
    schema_vector.push_back(arrow::field(std::to_string(i), arr_ptr->type()));
  }

  VLOG(0) << "ArrowWriter - Close - conversion to arrow arrays finished";


  // create schema from fields
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, arrays);

  VLOG(0) << "ArrowWriter - Close - table created";


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

  VLOG(0) << "ArrowWriter - Close - written table to file";
  tensor_data_.clear();
  tensors_.clear();
  return Status::OK();
}

// Assumption: tensors contains one row of the table.
Status ArrowWriter::WriteTensors(std::vector<Tensor> &tensors) {

  for(Tensor t : tensors) {
    // need to implicitly get tensor shapes (dimension sizes) as it is not passed to writer.
    // we only check for the shape in the first row of tensors handed to this function.
    if(!dims_initialized_) {
      InitDims(t);
    }

    if(arrow_dtypes_[current_col_idx_]->Equals(arrow::utf8())) {
      VLOG(0) << "Processing String Tensor:" << std::string(t.tensor_data().data(), t.tensor_data().length());
    }

    // accumulate buffers in correct column:
    tensor_data_[current_col_idx_].push_back(t.tensor_data().data());
    VLOG(0) << "ArrowWriter - WriteTensors - Added data_buffer to corresponding column " << current_col_idx_;
    current_col_idx_ = (current_col_idx_ + 1) % ncols_;

    // TODO: ugly solution, find better way to keep data_buf reference (shared_ptr for example).
    // this seems to copy all the data every time.
    tensors_.push_back(t);
  }
  return Status::OK();
}

/// \brief Takes tensor t as argument and appends shape information to local vector col_dims_[i] where
/// t is the i-th tensor handed to this function.
void ArrowWriter::InitDims(Tensor  &t) {
  if(col_dims_.size() < ncols_) { // we need more tensors to complete shape information
    std::vector<int> single_col_dims;
    for (int64_t dim_size : t.shape().dim_sizes()) {
      int val = (int) dim_size;
      single_col_dims.push_back(val);
    }
    col_dims_.push_back(single_col_dims);
  } else {  // all shapes known, don't need more information
    dims_initialized_ = true;
  }
}



} // namespace easl
} // namespace data
} // namespace tensorflow
