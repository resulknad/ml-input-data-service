//
// Created by simon on 24.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H

#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "tensorflow/core/framework/types.h"
#include "arrow_util.h"
#include "tensorflow/core/kernels/data/experimental/easl_service/service_cache_util.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {
namespace arrow_async_writer{



class ArrowAsyncWriter : public BoundedMemoryWriter {
public:
    explicit ArrowAsyncWriter(const int writer_count, const uint64 memory_threshold);

    // method used to insert data into deque_.
    void InsertData(const std::vector<Tensor>& tensors) override;

    // unuesed in TFRecord writer
    void FirstRowInfo(const std::vector<Tensor>& tensors) override;

    // creates an empty ElementOrEOF with eof set to true.
    std::unique_ptr<ElementOrEOF> CreateEOFToken() override;

    void WriterThread(Env *env, const std::string &shard_directory,
                      int writer_id, const std::string& compression,
                      const DataTypeVector& output_types, int64 version) override;

private:

    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;
    bool first_row_shape_set_ = false;
    std::vector<TensorShape> first_row_shape_;

    // use to choose betw. normal and experimental mode. Default experimental.
    const bool experimental_ = true;
};

} // namespace arrow_async_wirter
} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_ASYNC_WRITER_H
