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


struct TensorData {
    std::vector<std::vector<Tensor>> tensor_batch;
    size_t byte_count = 0;
    bool end_of_sequence = false;
};


class ArrowRoundRobinWriter : public MultiThreadedAsyncWriter {
public:
    explicit ArrowRoundRobinWriter(const int writer_count);

    void Write(const std::vector<Tensor>& tensors) override;

    Status WriterThread(Env* env, const std::string& shard_directory,
                        uint64 checkpoint_id, const std::string& compression,
                        int64 version, DataTypeVector output_types) override;

    Status ArrowWrite(const std::string& shard_directory, TensorData &dat, uint64 writer_id);

    void SignalEOF() override;

    ~ArrowRoundRobinWriter() override= default;

private:

    void ConsumeTensors(TensorData* dat_out, int writer_id);
    void PushCurrentBatch();

    // store tensors until a writer thread "consumes" them
    std::deque<TensorData> deque_ TF_GUARDED_BY(mu_);
    TensorData current_batch_;

    // arrow metadata
    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;
    std::vector<DataType> first_row_dtype_;
    std::vector<size_t> tensor_data_len_;

    // threshold used to control bytes in entire writer pipeline. Default or set via version (TODO)
    size_t thresh_ = 2e9;
    size_t max_batch_size_ = 1e8;  // wait for this amount of data until waking up writer -> set by constructor
    size_t available_row_capacity_ = 0;  // how many rows can be inserted without checking bytes_written?
    mutex mu_by_;
    size_t bytes_written_ = 0;  // guraded by mu_by_ --> bytes written out to disk by writers
    size_t bytes_received_ = 0;  // bytes received by iterator. Make sure bytes_received_ - bytes_written < thresh_
    condition_variable capacity_available_;  // if pipeline full, iterator thread waits until capacity available
    condition_variable tensors_available_;  // if pipeline empty, writer-threads sleep until tensors available
};

} // namespace arrow_round_robin
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ROUND_ROBIN_H
