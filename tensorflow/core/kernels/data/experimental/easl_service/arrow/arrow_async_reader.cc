//
// Created by simon on 24.04.21.
//

#include "arrow_async_reader.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_reader{

ArrowAsyncReader::ArrowAsyncReader(Env *env, const std::string &target_dir, const DataTypeVector &output_dtypes,
        const std::vector<PartialTensorShape> &output_shapes, int reader_count) :
        MultiThreadedAsyncReader(env, target_dir, output_dtypes, output_shapes, reader_count) {
  metadata_ = std::make_shared<ArrowUtil::ArrowMetadata>();
  metadata_->ReadMetadataFromFile(env, target_dir);

  #ifdef DEBUGGING
  VLOG(0) << "[ArrowAsyncReader] created.";
  #endif

}

Status ArrowAsyncReader::ReaderThread(
        Env *env, uint64 writer_id, int64 version,
        DataTypeVector output_types,
        std::vector<PartialTensorShape> output_shapes) {

  #ifdef DEBUGGING
  VLOG(0) << "[Thread " << writer_id << "] started running.";
  #endif
  metadata_->RegisterWorker();

  tensorflow::profiler::TraceMe activity(
          "EASLReaderThread", tensorflow::profiler::TraceMeLevel::kVerbose);

  bool end_of_sequence = false;

  while (!end_of_sequence) {
    std::string file_path;
    Consume(&file_path, &end_of_sequence);


    if (!end_of_sequence) {

      // create new arrow reader for each new file
      std::unique_ptr<ArrowReader> arrowReader;  // once out of scope -> destroyed
      arrowReader = absl::make_unique<ArrowReader>(col_selection_);
      // arrow reader gets re-initialized for every new file it is reading.
      arrowReader->Initialize(env, file_path, io::compression::kNone,
                              output_types, output_shapes, metadata_);

      #ifdef DEBUGGING
      int count = 0;
      VLOG(0) << "[Thread " << writer_id << "] initialized arrowReader";
      #endif

      bool eof = false;
      while (!eof) {
        #ifdef DEBUGGING
        VLOG(0) << "[Thread " << writer_id << "] reading tensors from RecordBatch...";
        #endif
        std::vector<Tensor> tensors;

        #ifdef STATS_LOG
        logger_->BeginWriteTensors(writer_id);
        #endif
        Status s = arrowReader->ReadTensors(&tensors);
        #ifdef STATS_LOG
        logger_->FinishWriteTensors(writer_id);
        #endif

        if (errors::IsOutOfRange(s)) {
          eof = true;  // can't break because of TFRecordReader.
          #ifdef DEBUGGING
          VLOG(0) << "[Thread " << writer_id << "] eof, fetching next file or exit...";
          #endif
        } else if(s != Status::OK()) {
          VLOG(0) << "Unexpected Error in ArrowAsyncReader, returning...";
          return s;
        }

        if(!tensors.empty()) {
          Add(tensors);
          #ifdef DEBUGGING
          VLOG(0) << "[Thread " << writer_id << "] tensors added to iterator. Total reads: " << ++count;
          #endif
        }
      }
    }
  }


  std::vector<Tensor> tensors;
  metadata_->GetLastRowBatch(&tensors);  // only returns something if it is last reader registered
  if(!tensors.empty()) {
    Add(tensors);
  }

  #ifdef DEBUGGING
  VLOG(0) << "[Thread " << writer_id << "] exit, notifying reader done";
  #endif

  mutex_lock l(mu_add_);
  num_readers_done_++;
  read_cv_.notify_one();
  return Status::OK();
}

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow