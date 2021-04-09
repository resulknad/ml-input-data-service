//
// Created by simon on 30.03.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_READER_H
#define ML_INPUT_DATA_SERVICE_ARROW_READER_H

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace data {
namespace easl{

class ArrowReader {
public:
    ArrowReader(Env *env, const std::string &filename,
                const string &compression_type,
                const DataTypeVector &dtypes);

    static void PrintTestLog();

    Status Initialize();

    Status ReadTensors(std::vector<Tensor> *read_tensors);

private:
    Status NextBatch();

    Env *env_;
    std::string filename_;
    string compression_type_;
    DataTypeVector dtypes_;

    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
    std::shared_ptr<arrow::RecordBatch> current_batch_;
    size_t current_batch_idx_;
    int64_t current_row_idx_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_READER_H