//
// Created by simon on 24.04.21.
//

#include "arrow_async_writer.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_writer.h"
#include "tensorflow/core/platform/stringprintf.h"

#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/service_cache.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"

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

ArrowAsyncWriter::ArrowAsyncWriter(const int writer_count) : MultiThreadedAsyncWriter(writer_count) {
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  metadata_->SetExperimental(experimental_);

}

Status ArrowAsyncWriter::WriterThread(Env* env, const std::string& shard_directory,
                    uint64 writer_id, const std::string& compression,
                    int64 version, DataTypeVector output_types) {

  uint64_t storageEstimate = 0; // estimated storage space on disk in bytes
  uint64 split_id = 0; // name all produced arrow files by this thread

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));

  // register thread for concurrently writing to arrowMetadata file
  metadata_->RegisterWorker();

  // create arrow writer
  std::unique_ptr<ArrowWriter> arrowWriter;
  arrowWriter = absl::make_unique<ArrowWriter>();

  // consume first tensor before creating arrow writer -> metadata needs row shape first
  snapshot_util::ElementOrEOF be;
  Consume(&be);
  TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id),
                                         compression, output_types, metadata_));

  while (true) {
    if (be.end_of_sequence) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      break;
    }

    storageEstimate += bytes_per_row_;

    // create new reader if memoryThreshold exceeded
    if(storageEstimate > memoryThreshold / writer_count_) {
      TF_RETURN_IF_ERROR(arrowWriter->Close());
      storageEstimate = bytes_per_row_; // reset storage estimate
      // create new writer for remaining tensors:
      arrowWriter = absl::make_unique<ArrowWriter>();
      TF_RETURN_IF_ERROR(arrowWriter->Create(env, GetFileName(shard_directory, writer_id, ++split_id),
                                             compression, output_types, metadata_));
    }

    TF_RETURN_IF_ERROR(arrowWriter->WriteTensors(be.value));

    // Consume tensors for next round
    Consume(&be);
  }

  // Write accumulated metadata before closing --> if last thread writes to file
  metadata_->WriteMetadataToFile(shard_directory);
  mutex_lock l(mu_);
  writer_finished_++;
  return Status::OK();
}

bool ArrowAsyncWriter::ProducerSpaceAvailable() {
  return (deque_.size() * bytes_per_row_) < producer_threshold_;
}

void ArrowAsyncWriter::Write(const std::vector<Tensor>& tensors) {
  if(!first_row_info_set_) {
    for(const Tensor& t : tensors) {
      bytes_per_row_ += t.TotalBytes();
      first_row_shape_.push_back(t.shape());
    }
    first_row_info_set_ = true;
  }
  metadata_->SetRowShape(first_row_shape_);
  mutex_lock l(mu_);
  mu_.Await( Condition(this, &ArrowAsyncWriter::ProducerSpaceAvailable));
  snapshot_util::ElementOrEOF element;
  element.value = tensors;
  deque_.push_back(std::move(element));
}

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow