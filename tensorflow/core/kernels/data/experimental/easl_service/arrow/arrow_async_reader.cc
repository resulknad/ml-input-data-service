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
}

Status ArrowAsyncReader::ReaderThread(
        Env *env, uint64 writer_id, int64 version,
        DataTypeVector output_types,
        std::vector<PartialTensorShape> output_shapes) {


  metadata_->RegisterWorker();

  tensorflow::profiler::TraceMe activity(
          "EASLReaderThread", tensorflow::profiler::TraceMeLevel::kVerbose);

  bool end_of_sequence = false;

  while (!end_of_sequence) {
    std::string file_path;
    Consume(&file_path, &end_of_sequence);

    if (!end_of_sequence) {

      std::unique_ptr<ArrowReader> arrowReader;

      arrowReader = absl::make_unique<ArrowReader>();
      Status s = arrowReader->Initialize(env, file_path, io::compression::kNone,
              output_types, output_shapes, metadata_);

      if(s != Status::OK()) {
        return s;
      }

      int64 count = 0;
      bool eof = false;
      while (!eof) {
        std::string t_str = "Reading Tensors:";
        std::vector<Tensor> tensors;
        Status s = arrowReader->ReadTensors(&tensors);
        if (errors::IsOutOfRange(s)) {
          eof = true;  // can't break because of TFRecordReader.
        } else if(s != Status::OK()) {
          return s;
        }

        if(!tensors.empty()) {
          Add(tensors);
        }
      }
    }
  }


  std::vector<Tensor> tensors;
  metadata_->GetLastRowBatch(&tensors);  // only returns something if it is last reader registered
  if(!tensors.empty()) {
    Add(tensors);
  }

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