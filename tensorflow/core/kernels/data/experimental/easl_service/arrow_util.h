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

/// \brief Construct a (nested) Arrow Array given a vector of flattened data buffers.
/// Each entry of the returned array corresponds to the data and shape of one tensor.
/// A batch of scalar tensors for example are represented as an array of arrow scalar values.
/// Supported primitive types for now: [Int8/16/32/64], [UInt8/16/32/64],
/// [HalfFloat], [Float], [Double]
///
/// \param[in] type Arrow Data type of the underlying primitive
/// used in the data buffer. Output Array will contain elements of this type.
/// \param[in] data Flattened data buffer. Assumes data stored in row major order.
/// \param[in] dim_size Sizes of all the dimensions
/// \param[in] out_array The array is returned via the shared pointer.
/// Caution: Don't forget to initialize the shared pointer before passing to this function!
Status GetArrayFromData(std::shared_ptr<arrow::DataType> type, std::vector<const char *>& data_column,
                        std::vector<int>& dim_size, std::shared_ptr<arrow::Array>* out_array);


/// \brief print content of a data buffer to string (binary representation).
/// prints memory words (4 bytes) in little-endian order.
/// starts with ptr[0], ptr[4], ...
std::string binaryToString(size_t size, const char* ptr);

} // namespace ArrowUtil
} //namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
