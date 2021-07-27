//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ASYNC_READER_H
#define ML_INPUT_DATA_SERVICE_ARROW_ASYNC_READER_H

#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/types.h"
#include "arrow_util.h"
#include "arrow_reader.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_reader{

using ArrowUtil::ArrowMetadata;

class ArrowAsyncReader : public MultiThreadedAsyncReader {
public:
    ArrowAsyncReader(Env *env,
                     std::shared_ptr<SplitProvider> split_provider,
                     const std::string &target_dir,
                     const DataTypeVector &output_dtypes,
                     const std::vector<PartialTensorShape> &output_shapes,
                     int reader_count = 8);

    Status ReaderThread(Env *env, uint64 writer_id, int64 version,
                        DataTypeVector output_types,
                        std::vector<PartialTensorShape> output_shapes);

private:
    std::shared_ptr<ArrowMetadata> metadata_;
};

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_READER_H
