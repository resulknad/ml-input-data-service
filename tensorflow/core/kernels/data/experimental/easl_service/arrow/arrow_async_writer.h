//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include <map>

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_writer{

class Metadata {
public:
    /// \brief write accumulated metadata to a file in serialized form
    Status WriteMetadata();

    /// \brief remembers which files contain partially filled batches at the end of
    /// the file. The last row of tensors stored in the arrays will have a different shapes.
    Status AddPartialBatch(string doc, std::vector<TensorShape> last_batch_shape) TF_LOCKS_EXCLUDED(mu_);

    /// \brief (general) shape of all dataset rows, one shape per dataset column. If
    /// batching is enabled, there may be tensors in the last row of the dataset that have a
    /// different shape and thus don't conform to this shape specification (see AddPartialBatch).
    Status SetRowShape(std::vector<TensorShape> row_shape);

private:
    mutex mu_;  // allow multiple threads to add values to Metadata File
    bool partial_batching_;
    std::vector<TensorShape> shapes_;
    std::map<string, std::vector<TensorShape>> partial_batch_shapes_; TF_GUARDED_BY(mu_);
};

class ArrowAsyncWriter : public MultiThreadedAsyncWriter {
public:
    ArrowAsyncWriter(const int writer_count);

    Status WriterThread(Env* env, const std::string& shard_directory,
                        uint64 checkpoint_id, const std::string& compression,
                        int64 version, DataTypeVector output_types);
private:
    const uint64 memoryThreshold = 1 << 28;
    Metadata metadata_;
};

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
