//
// Created by simon on 07.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_WRITER_H

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_util.h"
#include "tensorflow/core/framework/types.h"


namespace tensorflow {
namespace data {
namespace easl{

class ArrowWriter {
public:
    ArrowWriter();

    Status Create(Env *env, const string &filename,
                  const string &compression_type,
                  const DataTypeVector &dtypes,
                  std::shared_ptr<ArrowUtil::ArrowMetadata> metadata);

    Status Close();

    Status WriteTensors(std::vector<Tensor> &tensors);

    ~ArrowWriter()= default;

private:
    arrow::Compression::type getArrowCompressionType();

    std::shared_ptr<ArrowUtil::ArrowMetadata> metadata_;
    Env *env_;
    std::string filename_;
    string compression_type_;
    DataTypeVector dtypes_;
    arrow::DataTypeVector arrow_dtypes_;
    int32_t ncols_ = 0;
    int32_t current_col_idx_ = 0;

    std::vector<TensorShape> shapes_;
    std::vector<TensorShape> partial_shapes_;

    // initially false, true after first row has been read (implicitly get TensorShape)
    bool tensor_data_len_initialized_;

    std::vector<std::vector<const char *>> tensor_data_;
    std::vector<uint64> tensor_data_len_;
    // last tensor per column is usually smaller if batching enabled
    std::vector<uint64> last_row_len_;

    // memory allocator for string buffers
    Allocator* string_allocator_;

    // prevents memory pool from de-allocating tensor data buffs
    std::deque<Tensor> tensors_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow


#endif //ML_INPUT_DATA_SERVICE_ARROW_WRITER_H