//
// Created by simon on 30.03.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_READER_H
#define ML_INPUT_DATA_SERVICE_ARROW_READER_H

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace data {
namespace easl{

class ArrowReader {
public:
    ArrowReader();

    Status Initialize(Env *env, const std::string &filename,
                      const string &compression_type,
                      const DataTypeVector &dtypes,
                      const std::vector<PartialTensorShape> &shapes,
                      std::shared_ptr<ArrowUtil::ArrowMetadata> metadata);

    /// \brief Read an entire record batch into a vector<Tensor>.
    Status ReadTensors(std::vector<Tensor> *read_tensors);

    ~ArrowReader()= default;

private:
    /// \brief increments current_batch_idx_ by 1 (initialized to -1). If no more batches,
    /// return status with OUT_OF_RANGE error.
    Status NextBatch();

    /// \brief If no metadata provided for shapes / types, extract them implicitly from arrow arrays.
    /// Looks at first row across all RecordBatches to get the shapes / types of the first dataset row.
    Status InitShapesAndTypes();

    Env *env_;
    std::string filename_;
    string compression_type_;
    DataTypeVector dtypes_;
    std::vector<TensorShape> shapes_;

    // only used if batching (partially filled tensors)
    std::vector<TensorShape> partial_shapes;
    uint64_t total_rows_;
    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;


    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
    std::shared_ptr<arrow::RecordBatch> current_batch_;
    size_t current_batch_idx_;
    bool shapes_initialized_ = false;
    bool experimental_ = false;
    int64_t current_row_idx_;

    // used for column selection. hardcoded at the moment.
    std::vector<int> col_selection_ {5};  // only take one column for now. If empty return all.
    std::shared_ptr<arrow::ipc::RecordBatchFileReader> rfr_;
    std::shared_ptr<arrow::io::MemoryMappedFile> file_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_READER_H