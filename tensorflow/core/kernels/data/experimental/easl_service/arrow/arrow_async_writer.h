//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H

#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/types.h"
#include "arrow_util.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"
#include <map>

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_writer{



class ArrowAsyncWriter : public MultiThreadedAsyncWriter {
public:
    explicit ArrowAsyncWriter(const int writer_count);

    void Write(const std::vector<Tensor>& tensors) override TF_LOCKS_EXCLUDED(mu_);

    Status WriterThread(Env* env, const std::string& shard_directory,
                        uint64 checkpoint_id, const std::string& compression,
                        int64 version, DataTypeVector output_types) override;
private:

    bool ProducerSpaceAvailable() override TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

    const uint64 memoryThreshold = 1e9;  // 1 GB
    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;
    bool first_row_shape_set_ = false;

    // use to choose betw. normal and experimental mode. Default experimental.
    const bool experimental_ = true;
};

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
