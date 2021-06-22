//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ROUND_ROBIN_H
#define ML_INPUT_DATA_SERVICE_ARROW_ROUND_ROBIN_H

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
#include "arrow_util.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_round_robin {


struct BatchOrEOF : ElementOrEOF{
    std::vector<std::vector<Tensor>> tensor_batch;
    size_t byte_count = 0;
};


class ArrowRoundRobinWriter : public BoundedMemoryWriter {
public:
    explicit ArrowRoundRobinWriter(int writer_count, uint64 memory_threshold, int compression);

    // method used to insert data into deque_.
    void InsertData(const std::vector<Tensor>& tensors) override;

    // unuesed in TFRecord writer
    void FirstRowInfo(const std::vector<Tensor>& tensors) override;

    // creates an empty ElementOrEOF with eof set to true.
    std::unique_ptr<ElementOrEOF> CreateEOFToken() override;

    void WriterThread(Env *env, const std::string &shard_directory,
                      int writer_id, const std::string& compression, const DataTypeVector& output_types, int64 version) override;

    void Cleanup() override;

    std::shared_ptr<arrow::RecordBatch> RecordBatchFromTensorData(const std::string& filename, BatchOrEOF &dat);

private:
    arrow::ipc::IpcWriteOptions wo_ = {
            false,
            10,
            8,
            false,
            arrow::default_memory_pool(),
            arrow::util::Codec::Create(arrow::Compression::LZ4_FRAME).ValueOrDie(),
            false,
            false,
            arrow::ipc::MetadataVersion::V5
    };

    std::shared_ptr<arrow::Schema> schema_;
    std::unique_ptr<BatchOrEOF> current_batch_;  // batch we're currently filling

    // arrow metadata
    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;
    std::vector<DataType> first_row_dtype_;
    std::vector<TensorShape> first_row_shape_;
    std::vector<size_t> tensor_data_len_;

    size_t max_batch_size_;  // wait for this amount of data until waking up writer -> set by constructor
};

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ROUND_ROBIN_H
