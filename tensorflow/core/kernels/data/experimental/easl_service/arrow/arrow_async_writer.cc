//
// Created by simon on 24.04.21.
//

#include "arrow_async_writer.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_reader.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_writer.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_writer{

namespace {
  std::string GetFileName(const std::string& shard_directory,
                          uint64 file_id, uint64 split_id = 0) {
    return io::JoinPath(shard_directory, strings::Printf("%07llu_%llu.easl",
                                                         static_cast<unsigned long long>(file_id),
                                                         static_cast<unsigned long long>(split_id)));
  }
}

Status Metadata::WriteMetadata() {
  // TODO: serialization
}

Status Metadata::AddPartialBatch(string doc, std::vector<TensorShape> last_batch_shape) {
  mutex_lock l(mu_);  // unlocked automatically upon function return
  this->partial_batch_shapes_.insert({doc, last_batch_shape});
  this->partial_batching_ = true;
}

Status Metadata::SetRowShape(std::vector<TensorShape> row_shape) {
  this->shapes_ = row_shape;
}




ArrowAsyncWriter::ArrowAsyncWriter(const int writer_count) : MultiThreadedAsyncWriter(writer_count) {}

Status ArrowAsyncWriter::WriterThread(Env* env, const std::string& shard_directory,
                    uint64 writer_id, const std::string& compression,
                    int64 version, DataTypeVector output_types) {

  uint64_t storageEstimate = 0; // estimated storage space on disk in bytes
  uint64_t rowStorage = 0; // storage size of a single dataset row (single be.value). Assume all have the same size.
  uint64 split_id = 0; // name all produced arrow files by this thread

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));
  LOG(INFO) << "(Writer_" << writer_id << ") Created Dir ";

  std::unique_ptr<ArrowWriter> arrowWriter;

  arrowWriter = absl::make_unique<ArrowWriter>();
  TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id),
                                         compression, output_types));

  int count = 0;
  LOG(INFO) << "(Writer_" << writer_id << ") Starting to write ";

  while (true) {
    snapshot_util::ElementOrEOF be;
    Consume(&be);

    LOG(INFO) << "(Writer_" << writer_id << ") Read - "
              << be.end_of_sequence << " - Total: " << ++count;
    if (be.end_of_sequence) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      LOG(INFO) << "(Writer_" << writer_id << ") Closed w/ total read "
                << count;
      break;
    }

    // update memory estimate:
    if(rowStorage == 0) {
      std::vector<Tensor> &tensors = be.value;
      for(Tensor t : tensors) {
        rowStorage += t.TotalBytes();
      }
    } else {
      storageEstimate += rowStorage;
    }

    // create new reader if memoryThreshold exceeded
    if(storageEstimate > memoryThreshold) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      storageEstimate = rowStorage;
      // create new writer for remaining tensors:
      arrowWriter = absl::make_unique<ArrowWriter>();
      TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id, ++split_id),
                                             compression, output_types));
      LOG(INFO) << "(Writer_" << writer_id << ") Exceeded memory threshold, created new file (split_id = "
                                              "" << split_id <<")...";
    }

    TF_RETURN_IF_ERROR(arrowWriter->WriteTensors(be.value));
  }
  return Status::OK();
}

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow