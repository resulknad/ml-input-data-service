//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/profiler/lib/traceme.h"
#include "arrow_writer.h"
#include "arrow/api.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {


void ArrowWriter::PrintTestLog() {
  VLOG(0) << "ArrowWriter - TestLog\nArrow Version: " << arrow::GetBuildInfo().version_string;
}

ArrowWriter::ArrowWriter(Env *env, const std::string &filename,
                         const string &compression_type,
                         const DataTypeVector &dtypes) {
  this->env_ = env;
  this->filename_ = filename;
  this->compression_type_ = compression_type;
  this->dtypes_ = dtypes;
}

Status ArrowWriter::Close() {
  // TODO: build table from table-builders

  // build dummy table to test build
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  arrow::Int32Builder int_builder(pool);
  for(int i = 0; i < 5; i++) {
    CHECK_ARROW(int_builder.Append(i));
  }
  std::shared_ptr<arrow::Array> int_array;
  CHECK_ARROW(int_builder.Finish(&int_array));
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("int32", arrow::int32())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, {int_array});

  // write table to file:
  std::shared_ptr<arrow::io::FileOutputStream> file;
  ARROW_ASSIGN_CHECKED(file, arrow::io::FileOutputStream::Open(filename_, /*append=*/false));
  CHECK_ARROW(arrow::ipc::feather::WriteTable(*table, file.get(), arrow::ipc::feather::WriteProperties::Defaults()));
  CHECK_ARROW(file->Close());

  VLOG(0) << "ArrowWriter: written table to file";
  tensors_.clear();
  return Status::OK();
}

// Assumption: tensors contains one row of the table.
Status ArrowWriter::WriteTensors(std::vector<Tensor> &tensors) {
  for(Tensor t : tensors) {
    // check whether all needed information is in tensors:
    VLOG(0) << "ArrowWriter: TensorInfo ---- Shape: " << t.shape().DebugString() << "\t Type: " << DataTypeString(t.dtype());

    // print out tensor data:
    char* data_buf = (char *) t.tensor_data().data();
    typedef int64_t tensor_type;
    for(int i = 0; i < t.shape().num_elements(); i += 8) { //always alligned on 64-bit boundaries.
      VLOG(0) << "ArrowWriter: Tensor element " << i / sizeof(tensor_type) << ": \t" << (tensor_type) data_buf[i];
    }

    tensors_.push_back(t);
  }
  return Status::OK();
}

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow
