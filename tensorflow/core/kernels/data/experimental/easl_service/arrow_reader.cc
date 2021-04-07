//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/profiler/lib/traceme.h"
#include "arrow_reader.h"
#include "arrow/api.h"
#include "arrow/ipc/feather.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/io/file.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {


    void ArrowReader::PrintTestLog() {
      VLOG(0) << "ARROW - TestLog\nArrow Version: " << arrow::GetBuildInfo().version_string;
    }

    ArrowReader::ArrowReader(Env *env, const std::string &filename,
                         const string &compression_type,
                         const DataTypeVector &dtypes) {
      this->env_ = env;
      this->filename_ = filename;
      this->compression_type_ = compression_type;
      this->dtypes_ = dtypes;

    }

    Status ArrowReader::Initialize() {
      //TODO open file --> initialize stream
      std::shared_ptr<arrow::io::MemoryMappedFile> file;
      ARROW_ASSIGN_CHECKED(file, arrow::io::MemoryMappedFile::Open(filename_, arrow::io::FileMode::READ))
      std::shared_ptr<arrow::ipc::feather::Reader> reader;
      ARROW_ASSIGN_CHECKED(reader, arrow::ipc::feather::Reader::Open(file));
      std::shared_ptr<::arrow::Table> table;
      CHECK_ARROW(reader->Read(&table));

      int64_t n = table->num_columns();
      VLOG(0) << "ARROW: read table from file successfully. Num Columns: " << n;
      return Status::OK();
    }

    Status ArrowReader::ReadTensors(std::vector<Tensor> *read_tensors) {
      // reader reads one tensor and returns out of range.
      Tensor t = Tensor((int64) 0);
      read_tensors->push_back(t);
      t = Tensor((int64) 1);
      read_tensors->push_back(t);
      t = Tensor((int64) 2);
      read_tensors->push_back(t);
      t = Tensor((int64) 3);
      read_tensors->push_back(t);
      t = Tensor((int64) 4);
      read_tensors->push_back(t);

      Status s = Status(error::OUT_OF_RANGE, "dummy msg");
      return s;
    }

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow