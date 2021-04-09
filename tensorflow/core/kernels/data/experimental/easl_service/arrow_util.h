//
// Created by simon on 07.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
#define ML_INPUT_DATA_SERVICE_ARROW_UTIL_H

// dependencies ----------------------------------------------
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/kernels/data/experimental/snapshot_util.h"
#include "arrow/api.h"
#include "arrow/ipc/api.h"
#include "arrow/util/io_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"


// used macros -----------------------------------------------
#define CHECK_ARROW(arrow_status)             \
  do {                                        \
    arrow::Status _s = (arrow_status);        \
    if (!_s.ok()) {                           \
      return errors::Internal(_s.ToString()); \
    }                                         \
  } while (false)

#define ARROW_ASSIGN_CHECKED(lhs, rexpr)                                              \
  ARROW_ASSIGN_CHECKED_IMPL(ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), \
                             lhs, rexpr);

#define ARROW_ASSIGN_CHECKED_IMPL(result_name, lhs, rexpr) \
  auto&& result_name = (rexpr);                             \
  CHECK_ARROW((result_name).status());              \
  lhs = std::move(result_name).ValueUnsafe();

// Forward declaration  ---------------------------------------
namespace tensorflow {

class Tensor;
class TensorShape;

namespace data {
namespace easl {
namespace ArrowUtil {

// utility functions ------------------------------------------
// Convert Arrow Data Type to TensorFlow
Status GetTensorFlowType(std::shared_ptr<::arrow::DataType> dtype,
                         ::tensorflow::DataType* out);

// Convert TensorFlow Data Type to Arrow
Status GetArrowType(::tensorflow::DataType dtype,
                    std::shared_ptr<::arrow::DataType>* out);

// Assign equivalent TensorShape for the given Arrow Array
Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                   int64 batch_size, TensorShape* out_shape);

// Assign DataType and equivalent TensorShape for the given Arrow Array
Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                  int64 batch_size, ::tensorflow::DataType* out_dtype,
                  TensorShape* out_shape);

// Assign elements of an Arrow Array to a Tensor
Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                    Tensor* out_tensor);

} // namespace ArrowUtil
} //namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
