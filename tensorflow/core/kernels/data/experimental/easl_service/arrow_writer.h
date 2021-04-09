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
    ArrowWriter(Env *env, const string &filename,
                const string &compression_type,
                const DataTypeVector &dtypes);

    static void PrintTestLog();

    Status Close();

    Status WriteTensors(std::vector<Tensor> &tensors);

private:
    Env *env_;
    std::string filename_;
    string compression_type_;
    DataTypeVector dtypes_;
    std::deque<Tensor> tensors_;
};

} // namespace easl
} // namespace data
} // namespace tensorflow


#endif //ML_INPUT_DATA_SERVICE_ARROW_WRITER_H