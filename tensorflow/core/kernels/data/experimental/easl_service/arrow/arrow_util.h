//
// Created by simon on 07.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
#define ML_INPUT_DATA_SERVICE_ARROW_UTIL_H

// dependencies ----------------------------------------------
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/data/snapshot_utils.h"
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


class ArrowMetadata {
public:
    ArrowMetadata();

    /// \brief write accumulated metadata to a file in serialized form
    Status WriteMetadataToFile(const std::string& path) TF_LOCKS_EXCLUDED(mu_);

    /// \brief read and deserialize metadata from file
    Status ReadMetadataFromFile(Env* env, const std::string& path);

    /// \brief remembers which files contain partially filled batches at the end of
    /// the file. The last row of tensors stored in the arrays will have a different shapes.
    Status AddPartialBatch(const string& doc, const std::vector<TensorShape>& last_batch_shape) TF_LOCKS_EXCLUDED(mu_);

    Status GetPartialBatches(string doc, std::vector<TensorShape>* out_last_batch_shape);

    bool IsPartialBatching();

    Status GetRowShape(std::vector<TensorShape>* out_row_shape);

    /// \brief (general) shape of all dataset rows, one shape per dataset column. If
    /// batching is enabled, there may be tensors in the last row of the dataset that have a
    /// different shape and thus don't conform to this shape specification (see AddPartialBatch).
    Status SetRowShape(std::vector<TensorShape> row_shape) TF_LOCKS_EXCLUDED(mu_);

    Status RegisterWorker();

    /// \brief specifies whether experimental more efficient data storage format is used for reading
    bool IsExperimental();

    Status SetExperimental(bool exp);

    // Assumption: only one file contains partial batches (end of dataset)
    Status AddLastRowBatch(Tensor &t);

    // only returns last row tensors if we are at the end of the dataset
    Status GetLastRowBatch(std::vector<Tensor> *out) TF_LOCKS_EXCLUDED(mu_);

private:
    Status WriteData(const std::string& path);

    std::vector<Tensor> last_row_batches_;  // used to pass partially filled last row batches to async_reader
    mutex mu_;  // allow multiple threads to add values to Metadata File
    bool partial_batching_ = false;
    bool experimental_ = true;
    std::vector<TensorShape> shapes_;
    int num_worker_threads_ TF_GUARDED_BY(mu_); // num writer_threads still actively writing
    std::map<string, std::vector<TensorShape>> partial_batch_shapes_ TF_GUARDED_BY(mu_);
};



// utility functions ------------------------------------------
// Convert Arrow Data Type to TensorFlow
Status GetTensorFlowType(const std::shared_ptr<::arrow::DataType>& dtype,
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
arrow::Status GetArrayFromData(std::shared_ptr<arrow::DataType> type, std::vector<const char *>& data_column,
                        const absl::InlinedVector<int64 , 4>& dim_size, std::shared_ptr<arrow::Array>* out_array,
                        const absl::InlinedVector<int64, 4>& last_dim_size);


/// \brief experimental ultra fast writer for non-complex data types (i.e. string not supported)
arrow::Status GetArrayFromDataExperimental(
        size_t buff_len,
        std::vector<const char *>& data_column,
        std::shared_ptr<arrow::Array>* out_array,
        int64 last_buff_len);


/// \brief experimental ultra fast reader for non-complex data types (i.e. string not supported)
Status AssignTensorExperimental(
        std::shared_ptr<arrow::Array> array,
        int64 i,
        Tensor* out_tensor);


/// \brief print content of a data buffer to string (binary representation).
/// prints memory words (4 bytes) in little-endian order.
/// starts with ptr[0], ptr[4], ...
std::string binaryToString(size_t size, const char* ptr);

} // namespace ArrowUtil
} //namespace easl
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
