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

ArrowAsyncWriter::ArrowAsyncWriter(const int writer_count, const uint64 memory_threshold)
  : BoundedMemoryWriter(writer_count, memory_threshold) {
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  metadata_->SetExperimental(experimental_);
}

void ArrowAsyncWriter::WriterThread(Env *env, const std::string &shard_directory,
                                    int writer_id, const std::string& compression,
                                    const DataTypeVector& output_types, int64 version) {

  uint64_t storageEstimate = 0; // estimated storage space on disk in bytes
  uint64 split_id = 0; // name all produced arrow files by this thread

  env->RecursivelyCreateDir(shard_directory);

  // register thread for concurrently writing to arrowMetadata file
  metadata_->RegisterWorker();

  // create arrow writer
  std::unique_ptr<ArrowWriter> arrowWriter;
  arrowWriter = absl::make_unique<ArrowWriter>();

  // consume first tensor before creating arrow writer -> metadata needs row shape first
  std::unique_ptr<ElementOrEOF> parent_be = Consume(writer_id);

  arrowWriter->Create(env, GetFileName(shard_directory, writer_id),
          compression, output_types, metadata_);

  while (true) {
    auto* be = dynamic_cast<RowOrEOF*>(parent_be.get());

    if (be->eof) {
      arrowWriter->Close();
      break;
    }

    storageEstimate += bytes_per_row_;

    #ifdef DEBUGGING
    VLOG(0) << "ArrowAsyncWriter - Storage Estimate: " << storageEstimate << " bpr: " << bytes_per_row_;
    #endif

    // create new reader if memoryThreshold exceeded
    if(storageEstimate >= memory_threshold_ / writer_count_) {
      arrowWriter->Close();
      storageEstimate = bytes_per_row_; // reset storage estimate
      // create new writer for remaining tensors:
      arrowWriter = absl::make_unique<ArrowWriter>();
      arrowWriter->Create(env, GetFileName(shard_directory, writer_id, ++split_id),
              compression, output_types, metadata_);
    }

    arrowWriter->WriteTensors(be->data);

    // Consume tensors for next round
    parent_be = Consume(writer_id);
  }

  // Write accumulated metadata before closing --> if last thread writes to file
  metadata_->WriteMetadataToFile(shard_directory);
  mutex_lock l(mu_);
  WriterReturn(writer_id);
}

void ArrowAsyncWriter::FirstRowInfo(const std::vector<Tensor> &tensors) {
  for(const Tensor& t : tensors) {
    first_row_shape_.push_back(t.shape());
  }
  metadata_->SetRowShape(first_row_shape_);
}

void ArrowAsyncWriter::InsertData(const std::vector<Tensor> &tensors) {
  mutex_lock l(mu_);
  std::unique_ptr<RowOrEOF> r_dat = absl::make_unique<RowOrEOF>();
  r_dat->eof = false;
  r_dat->data = tensors;
  deque_.push_back(std::move(r_dat));
}

std::unique_ptr<ElementOrEOF> ArrowAsyncWriter::CreateEOFToken() {
  std::unique_ptr<RowOrEOF> r_eof = absl::make_unique<RowOrEOF>();
  r_eof->eof = true;
  return std::move(r_eof);
}


} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow