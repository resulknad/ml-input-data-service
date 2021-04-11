//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_writer.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"

namespace tensorflow {
namespace data {
namespace easl{

void ArrowWriter::PrintTestLog() {
  VLOG(0) << "ArrowWriter - PrintTestLog: " << arrow::GetBuildInfo().version_string;
}

ArrowWriter::ArrowWriter() {} // empty constructor, call Create for error handling.

Status ArrowWriter::Create(Env *env, const std::string &filename,
                         const string &compression_type,
                         const DataTypeVector &dtypes) {
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
  this->ncols_ = dtypes.size(); // TODO: make sure this always corresponds to number of dtypes!!
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
    std::vector<char *> col;
    tensor_data_.push_back(col);
  }

  VLOG(0) << "ArrowWriter - Create - Initialized with the following parameters:\n"
             "Filename: " << filename_ << "\nCompression_Type: " << compression_type_ << "\n"
             "DataTypes: " << DataTypeVectorString(dtypes_) << "\nncols: " << ncols_ << "\n"
             "ArrowDataTypes[0]: " << arrow_dtypes_[0]->ToString();
}


// build dummy table to test with fixed schema
Status BuildExampleTable(std::shared_ptr<arrow::Table>& table) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();

  arrow::Int32Builder int1_builder(pool);
  arrow::Int32Builder int2_builder(pool);

  for(int i = 0; i < 5; i++) {
    CHECK_ARROW(int1_builder.Append(i));
    CHECK_ARROW(int2_builder.Append(i * i));

  }
  std::shared_ptr<arrow::Array> int1_array;
  std::shared_ptr<arrow::Array> int2_array;

  CHECK_ARROW(int1_builder.Finish(&int1_array));
  CHECK_ARROW(int2_builder.Finish(&int2_array));

  std::vector<std::shared_ptr<arrow::Field>> schema_vector =
          {arrow::field("int1", arrow::int32()), arrow::field("int2", arrow::int32())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  table = arrow::Table::Make(schema, {int1_array, int2_array});
  return Status::OK();
}


Status ArrowWriter::Close() {

  VLOG(0) << "ArrowWriter - Close - invoked";

  // get converted arrow array for each column:
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  for(int i = 0; i < ncols_; i++) {
    std::shared_ptr<arrow::Array> arr_ptr;
    VLOG(0) << "ArrowWriter - Close - converting " << i << "th column to array";

    TF_RETURN_IF_ERROR(ArrowUtil::GetArrayFromData(arrow_dtypes_[i], tensor_data_[i], col_dims_[i], &arr_ptr));
    VLOG(0) << "ArrowWriter - Close - conversion completed: \n "
                     "" << arr_ptr->ToString();

    arrays.push_back(arr_ptr);
    schema_vector.push_back(arrow::field(""+i, arr_ptr->type()));
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
  CHECK_ARROW(arrow::ipc::feather::WriteTable(*table, file.get(), arrow::ipc::feather::WriteProperties::Defaults()));
  CHECK_ARROW(file->Close());

  VLOG(0) << "ArrowWriter - Close - written table to file";
  tensor_data_.clear();
  return Status::OK();
}

// Assumption: tensors contains one row of the table.
Status ArrowWriter::WriteTensors(std::vector<Tensor> &tensors) {

  for(Tensor t : tensors) {
    if(!dims_initialized_) {
      InitDims(t);
    }


    // check whether all needed information is in tensors:
    VLOG(0) << "ArrowWriter - WriteTensors - TensorInfo ---- Shape: " << t.shape().DebugString() << ""
                     "\t Type: " << DataTypeString(t.dtype()) << "\t NumEle: " << t.shape().num_elements() << ""
                     "\nDims: " << t.shape().dims() << " \nDimension Sizes: Todo";


    // accumulate buffers in correct column:
    char* data_buf = (char *) t.tensor_data().data();
    tensor_data_[current_col_idx_].push_back(data_buf);
    VLOG(0) << "ArrowWriter - WriteTensors - Added data_buffer to corresponding column " << current_col_idx_;
    current_col_idx_ = (current_col_idx_ + 1) % ncols_;

    // TODO: maybe track estimated memory usage and flush to file before using too much
  }
  return Status::OK();
}

void ArrowWriter::InitDims(Tensor  &t) {
  if(col_dims_.size() < ncols_) {
    // TODO: not sure whether this works. test whether loop gets called for scalar shapes.
    std::string debug_string = "Dimensions:";

    std::vector<int> single_col_dims;
    for (int64_t dim_size : t.shape().dim_sizes()) {
      int val = (int) dim_size;
      single_col_dims.push_back(val);
      debug_string += "\t" + std::to_string(val);
    }
    VLOG(0) << "ArrowWriter - InitDims - " << debug_string;
    col_dims_.push_back(single_col_dims);
  } else {
    VLOG(0) << "ArrowWriter - InitDims - set dims_initialized to true";
    VLOG(0) << "ArrowWriter - InitDims - all dimensions:";
    for(int i = 0; i < col_dims_.size(); i++) {
      for(int j = 0; j < col_dims_[i].size(); j++) {
        VLOG(0) << "\tArrowWriter - InitDims - Column = " << i << " Dimension = " << j << " Val = " << col_dims_[i][j];
      }
    }

    dims_initialized_ = true;
  }
}

} // namespace easl
} // namespace data
} // namespace tensorflow
