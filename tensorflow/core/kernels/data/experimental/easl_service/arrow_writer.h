//
// Created by simon on 07.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_WRITER_H
#define ML_INPUT_DATA_SERVICE_ARROW_WRITER_H

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"
#include "tensorflow/core/framework/types.h"


namespace tensorflow {
namespace data {
namespace easl{

class ArrowWriter {
public:
    ArrowWriter();

    Status Create(Env *env, const string &filename,
                  const string &compression_type,
                  const DataTypeVector &dtypes);

    Status Close();

    Status WriteTensors(std::vector<Tensor> &tensors);

private:
    void InitDims(Tensor &t);
    arrow::Compression::type getArrowCompressionType();

    Env *env_;
    std::string filename_;
    string compression_type_;
    DataTypeVector dtypes_;
    arrow::DataTypeVector arrow_dtypes_;
    int32_t ncols_;
    int32_t current_col_idx_;

    // initially false, true after first row has been read (implicitly get TensorShape)
    bool dims_initialized_;
    std::vector<std::vector<int>> col_dims_;
    std::vector<std::vector<const char *>> tensor_data_;

    // memory allocator for string buffers
    Allocator* string_allocator_;

    // prevents memory pool from de-allocating tensor data buffs
    std::deque<Tensor> tensors_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow


#endif //ML_INPUT_DATA_SERVICE_ARROW_WRITER_H