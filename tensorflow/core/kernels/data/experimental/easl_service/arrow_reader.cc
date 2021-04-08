//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/profiler/lib/traceme.h"
#include "arrow_reader.h"
#include "arrow/api.h"
#include "arrow/ipc/feather.h"
#include "arrow/io/file.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {


    void ArrowReader::PrintTestLog() {
      VLOG(0) << "ARROW - TestLog\nArrow Version: " << arrow::GetBuildInfo().version_string;
    }

    ArrowReader::ArrowReader(Env *env, const std::string &filename,
                         const string &compression_type, const DataTypeVector &dtypes)
         : env_(env), filename_(filename), compression_type_(compression_type), dtypes_(dtypes){

      // initialize internal data structures
      this->current_batch_idx_ = -1;
    }

    Status ArrowReader::Initialize() {
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

      VLOG(0) << "ArrowReader - Initialize - read table into recordbatches.\n"
                 "Num record batches: " << record_batches_.size();
      return Status::OK();
    }

    Status ArrowReader::ReadTensors(std::vector<Tensor> *read_tensors) {

      // dummy reader for testing --------------
      Tensor t = Tensor((int32) current_batch_idx_);
      read_tensors->push_back(t);
      // ---------------------------------------

      TF_RETURN_IF_ERROR(NextBatch());
      // Invariant: current_batch_ != nullptr
      VLOG(0) << "ArrowReader - ReadTensors - at this point current_batch should never be null. -> " << (current_batch_ != nullptr);


      // logging information of record batches:
      VLOG(0) << "ArrowReader - ReadTensors - info of current_batch:\n"
                  "Schema : " << current_batch_->schema()->ToString() << "\n"
                  "NumColumns : " << current_batch_->num_columns() << "\n"
                  "NumRows : " << current_batch_->num_rows() << "\n"
                  "RecordBatch : " << current_batch_->ToString() << "\n";

      //TODO yields segmentation fault!
      for(int i; i < current_batch_->num_columns(); i++) {
        VLOG(0) << "Is this being called?";
//        std::shared_ptr<arrow::Array> arr = current_batch_->column(i);
//        VLOG(0) << "ArrowReader - ReadTensors - Reading column " << i << "of current_batch_:\n"
//                         "Is arr nullptr? -- " << (arr == nullptr);
        // TODO: go over all columns and append row-wise to read_tensors
        // TODO: convert to tensor
      }

      return Status::OK();
    }

    Status ArrowReader::NextBatch() {
      if (++current_batch_idx_ < record_batches_.size()) {
        current_batch_ = record_batches_[current_batch_idx_];
      } else  {
        return Status(error::OUT_OF_RANGE, "finished reading all record batches");
      }
      return Status::OK();
    }

} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow