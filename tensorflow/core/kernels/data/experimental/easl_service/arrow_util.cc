/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

More or less untouched code, adapted slightly for specific use case by simonsom
==============================================================================*/

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow_util.h"
#include "arrow/adapters/tensorflow/convert.h"
#include "arrow/api.h"
#include "arrow/ipc/api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace easl {
namespace ArrowUtil {

Status GetTensorFlowType(std::shared_ptr<::arrow::DataType> dtype,
                         ::tensorflow::DataType* out) {
  if (dtype->id() == ::arrow::Type::STRING) {
    *out = ::tensorflow::DT_STRING;
    return Status::OK();
  }
  ::arrow::Status status =
          ::arrow::adapters::tensorflow::GetTensorFlowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("arrow data type ", dtype,
                                   " is not supported: ", status);
  }
  return Status::OK();
}

Status GetArrowType(::tensorflow::DataType dtype,
                    std::shared_ptr<::arrow::DataType>* out) {
  if (dtype == ::tensorflow::DT_STRING) {
    *out = ::arrow::utf8();
    return Status::OK();
  }
  ::arrow::Status status =
          ::arrow::adapters::tensorflow::GetArrowType(dtype, out);
  if (!status.ok()) {
    return errors::InvalidArgument("tensorflow data type ", dtype,
                                   " is not supported: ", status);
  }
  return Status::OK();
}

class ArrowAssignSpecImpl : public arrow::ArrayVisitor {
public:
    ArrowAssignSpecImpl() : i_(0), batch_size_(0) {}

    Status AssignDataType(std::shared_ptr<arrow::Array> array,
                          ::tensorflow::DataType* out_dtype) {
      return AssignSpec(array, 0, 0, out_dtype, nullptr);
    }

    Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                       int64 batch_size, TensorShape* out_shape) {
      return AssignSpec(array, i, batch_size, nullptr, out_shape);
    }

    // Get the DataType and equivalent TensorShape for a given Array, taking into
    // account possible batch size
    Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                      int64 batch_size, ::tensorflow::DataType* out_dtype,
                      TensorShape* out_shape) {
      VLOG(0) << "ArrowUtil - AssignSpecImpl - AssignSpec - Invoked";
      i_ = i;
      batch_size_ = batch_size;
      out_shape_ = out_shape;
      out_dtype_ = out_dtype;

      // batch_size of 0 indicates 1 record at a time, no batching
      if (batch_size_ > 0) {
        out_shape_->AddDim(batch_size_);
      }

      CHECK_ARROW(array->Accept(this));
      return Status::OK();
    }

protected:
    template <typename ArrayType>
    arrow::Status VisitPrimitive(const ArrayType& array) {
      if (out_dtype_ != nullptr) {
        return ::arrow::adapters::tensorflow::GetTensorFlowType(array.type(),
                                                                out_dtype_);
      }
      return arrow::Status::OK();
    }

#define VISIT_PRIMITIVE(TYPE)                               \
  virtual arrow::Status Visit(const TYPE& array) override { \
    return VisitPrimitive(array);                           \
  }

  VISIT_PRIMITIVE(arrow::BooleanArray)
  VISIT_PRIMITIVE(arrow::Int8Array)
  VISIT_PRIMITIVE(arrow::Int16Array)
  VISIT_PRIMITIVE(arrow::Int32Array)
  VISIT_PRIMITIVE(arrow::Int64Array)
  VISIT_PRIMITIVE(arrow::UInt8Array)
  VISIT_PRIMITIVE(arrow::UInt16Array)
  VISIT_PRIMITIVE(arrow::UInt32Array)
  VISIT_PRIMITIVE(arrow::UInt64Array)
  VISIT_PRIMITIVE(arrow::HalfFloatArray)
  VISIT_PRIMITIVE(arrow::FloatArray)
  VISIT_PRIMITIVE(arrow::DoubleArray)
  VISIT_PRIMITIVE(arrow::StringArray)
#undef VISIT_PRIMITIVE

    virtual arrow::Status Visit(const arrow::ListArray& array) override {
      int32 values_offset = array.value_offset(i_);
      int32 array_length = array.value_length(i_);

      // what is this for? --> probably number of dimensions?
      int32 num_arrays = 2;

      VLOG(0) << "ArrowUtil - AssignSpecImpl - Visit(ListArray) - Invoked";

      // If batching tensors, arrays must be same length
      if (batch_size_ > 0) {
        num_arrays = batch_size_;
        for (int64 j = i_; j < i_ + num_arrays; ++j) {
          if (array.value_length(j) != array_length) {
            return arrow::Status::Invalid(
                    "Batching variable-length arrays is unsupported");
          }
        }
      }

      // Add diminsion for array
      if (out_shape_ != nullptr) {
        VLOG(0) << "ArrowUtil - AssignSpecImpl - Visit(ListArray) - add dimension for array."
                   "\n Adding Dimension Size: " << array_length;
        out_shape_->AddDim(array_length);
      }

      // Prepare the array data buffer and visit the array slice
      std::shared_ptr<arrow::Array> values = array.values();
      std::shared_ptr<arrow::Array> element_values =
              values->Slice(values_offset, array_length * num_arrays);
      return element_values->Accept(this);
    }

private:
    int64 i_;
    int64 batch_size_;
    DataType* out_dtype_;
    TensorShape* out_shape_;
};

Status AssignShape(std::shared_ptr<arrow::Array> array, int64 i,
                   int64 batch_size, TensorShape* out_shape) {
  ArrowAssignSpecImpl visitor;
  return visitor.AssignShape(array, i, batch_size, out_shape);
}

Status AssignSpec(std::shared_ptr<arrow::Array> array, int64 i,
                  int64 batch_size, ::tensorflow::DataType* out_dtype,
                  TensorShape* out_shape) {
  ArrowAssignSpecImpl visitor;
  return visitor.AssignSpec(array, i, batch_size, out_dtype, out_shape);
}

// Assign elements of an Arrow Array to a Tensor
class ArrowAssignTensorImpl : public arrow::ArrayVisitor {
public:
    ArrowAssignTensorImpl() : i_(0), out_tensor_(nullptr) {}

    Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                        Tensor* out_tensor) {
      VLOG(0) << "ArrowUtil - ArrowAssignTensorImpl - AssignTensor - Invoked";
      i_ = i;
      out_tensor_ = out_tensor;
      if (array->null_count() != 0) {
        return errors::Internal(
                "Arrow arrays with null values not currently supported");
      }
      CHECK_ARROW(array->Accept(this));
      return Status::OK();
    }

protected:
    virtual arrow::Status Visit(const arrow::BooleanArray& array) {
      // Must copy one value at a time because Arrow stores values as bits
      auto shape = out_tensor_->shape();
      for (int64 j = 0; j < shape.num_elements(); ++j) {
        // NOTE: for Array ListArray, curr_row_idx_ is 0 for element array
        bool value = array.Value(i_ + j);
        void* dst = const_cast<char*>(out_tensor_->tensor_data().data()) +
                    j * sizeof(value);
        memcpy(dst, &value, sizeof(value));
      }

      return arrow::Status::OK();
    }

    template <typename ArrayType>
    arrow::Status VisitFixedWidth(const ArrayType& array) {

      VLOG(0) << "ArrowUtil - ArrowAssignTensorImpl - VisitFixedWidth - Invoked\n"
                 "ArrayType: " << array.type()->ToString() << "\n"
                 "ArrayContent: " << array.ToString();

      const auto& fw_type =
              static_cast<const arrow::FixedWidthType&>(*array.type());
      const int64_t type_width = fw_type.bit_width() / 8;

      // TODO: verify tensor is correct shape, arrow array is within bounds

      // Primitive Arrow arrays have validity and value buffers, currently
      // only arrays with null count == 0 are supported, so only need values here
      static const int VALUE_BUFFER = 1;
      auto values = array.data()->buffers[VALUE_BUFFER];
      if (values == NULLPTR) {
        return arrow::Status::Invalid(
                "Received an Arrow array with a NULL value buffer");
      }

      const void* src =
              (values->data() + array.data()->offset * type_width) + i_ * type_width;
      void* dst = const_cast<char*>(out_tensor_->tensor_data().data());
      std::memcpy(dst, src, out_tensor_->NumElements() * type_width);

      return arrow::Status::OK();
    }

#define VISIT_FIXED_WIDTH(TYPE)                             \
virtual arrow::Status Visit(const TYPE& array) override { \
return VisitFixedWidth(array);                          \
}

    VISIT_FIXED_WIDTH(arrow::Int8Array)
    VISIT_FIXED_WIDTH(arrow::Int16Array)
    VISIT_FIXED_WIDTH(arrow::Int32Array)
    VISIT_FIXED_WIDTH(arrow::Int64Array)
    VISIT_FIXED_WIDTH(arrow::UInt8Array)
    VISIT_FIXED_WIDTH(arrow::UInt16Array)
    VISIT_FIXED_WIDTH(arrow::UInt32Array)
    VISIT_FIXED_WIDTH(arrow::UInt64Array)
    VISIT_FIXED_WIDTH(arrow::HalfFloatArray)
    VISIT_FIXED_WIDTH(arrow::FloatArray)
    VISIT_FIXED_WIDTH(arrow::DoubleArray)
#undef VISIT_FIXED_WITH

    virtual arrow::Status Visit(const arrow::ListArray& array) override {
      VLOG(0) << "ArrowUtil - ArrowAssignTensorImpl - Visit(ListArray) - Invoked";

      int32 values_offset = array.value_offset(i_);
      int32 curr_array_length = array.value_length(i_);
      int32 num_arrays = 1;
      auto shape = out_tensor_->shape();

      // If batching tensors, arrays must be same length
      if (shape.dims() > 1) {
        num_arrays = shape.dim_size(0);
        for (int64_t j = i_; j < i_ + num_arrays; ++j) {
          if (array.value_length(j) != curr_array_length) {
            return arrow::Status::Invalid(
                    "Batching variable-length arrays is unsupported");
          }
        }
      }

      // Save current index and swap after array is copied
      int32 tmp_index = i_;
      i_ = 0;

      // Prepare the array data buffer and visit the array slice
      std::shared_ptr<arrow::Array> values = array.values();
      std::shared_ptr<arrow::Array> element_values =
              values->Slice(values_offset, curr_array_length * num_arrays);
      auto result = element_values->Accept(this);

      // Reset state variables for next time
      i_ = tmp_index;
      return result;
    }

    virtual arrow::Status Visit(const arrow::StringArray& array) override {
      if (!array.IsNull(i_)) {
        out_tensor_->scalar<tstring>()() = array.GetString(i_);
      } else {
        out_tensor_->scalar<tstring>()() = "";
      }
      return arrow::Status::OK();
    }

private:
    int64 i_;
    int32 curr_array_length_;
    Tensor* out_tensor_;
};

Status AssignTensor(std::shared_ptr<arrow::Array> array, int64 i,
                    Tensor* out_tensor) {
  ArrowAssignTensorImpl visitor;
  return visitor.AssignTensor(array, i, out_tensor);
}

// Check the type of an Arrow array matches expected tensor type
class ArrowArrayTypeCheckerImpl : public arrow::TypeVisitor {
  public:
      Status CheckArrayType(std::shared_ptr<arrow::DataType> type,
                            ::tensorflow::DataType expected_type) {
        expected_type_ = expected_type;

        // First see if complex type handled by visitor
        arrow::Status visit_status = type->Accept(this);
        if (visit_status.ok()) {
          return Status::OK();
        }

        // Check type as a scalar type
        CHECK_ARROW(CheckScalarType(type));
        return Status::OK();
      }

  protected:
      virtual arrow::Status Visit(const arrow::ListType& type) {
        return CheckScalarType(type.value_type());
      }

      // Check scalar types with arrow::adapters::tensorflow
      arrow::Status CheckScalarType(std::shared_ptr<arrow::DataType> scalar_type) {
        DataType converted_type;
        ::tensorflow::Status status =
                GetTensorFlowType(scalar_type, &converted_type);
        if (!status.ok()) {
          return ::arrow::Status::Invalid(status);
        }
        if (converted_type != expected_type_) {
          return arrow::Status::TypeError(
                  "Arrow type mismatch: expected dtype=" +
                  std::to_string(expected_type_) +
                  ", but got dtype=" + std::to_string(converted_type));
        }
        return arrow::Status::OK();
      }

  private:
      DataType expected_type_;
  };

  Status CheckArrayType(std::shared_ptr<arrow::DataType> type,
                        ::tensorflow::DataType expected_type) {
    ArrowArrayTypeCheckerImpl visitor;
    return visitor.CheckArrayType(type, expected_type);
  }

class ArrowMakeArrayDataImpl : public arrow::TypeVisitor {
  public:
      Status Make(std::shared_ptr<arrow::DataType> type,
                  std::vector<int64> array_lengths,
                  std::vector<std::shared_ptr<arrow::Buffer>> buffers,
                  std::shared_ptr<arrow::ArrayData>* out_data) {
        type_ = type;
        lengths_ = array_lengths;
        buffers_ = buffers;
        out_data_ = out_data;
        CHECK_ARROW(type->Accept(this));
        return Status::OK();
      }

  protected:
      template <typename DataTypeType>
      arrow::Status VisitPrimitive(const DataTypeType& type) {
        // TODO null count == 0
        *out_data_ =
                arrow::ArrayData::Make(type_, lengths_[0], std::move(buffers_), 0);
        return arrow::Status::OK();
      }

#define VISIT_PRIMITIVE(TYPE)                              \
  virtual arrow::Status Visit(const TYPE& type) override { \
    return VisitPrimitive(type);                           \
  }

  VISIT_PRIMITIVE(arrow::BooleanType)
  VISIT_PRIMITIVE(arrow::Int8Type)
  VISIT_PRIMITIVE(arrow::Int16Type)
  VISIT_PRIMITIVE(arrow::Int32Type)
  VISIT_PRIMITIVE(arrow::Int64Type)
  VISIT_PRIMITIVE(arrow::UInt8Type)
  VISIT_PRIMITIVE(arrow::UInt16Type)
  VISIT_PRIMITIVE(arrow::UInt32Type)
  VISIT_PRIMITIVE(arrow::UInt64Type)
  VISIT_PRIMITIVE(arrow::HalfFloatType)
  VISIT_PRIMITIVE(arrow::FloatType)
  VISIT_PRIMITIVE(arrow::DoubleType)
#undef VISIT_PRIMITIVE

  virtual arrow::Status Visit(const arrow::ListType& type) override {
    // TODO assert buffers and lengths size

    // Copy first 2 buffers, which are validity and offset buffers for the list
    std::vector<std::shared_ptr<arrow::Buffer>> list_bufs(buffers_.begin(),
                                                          buffers_.begin() + 2);
    buffers_.erase(buffers_.begin(), buffers_.begin() + 2);

    // Copy first array length for length of list
    int64 list_length = lengths_[0];
    lengths_.erase(lengths_.begin(), lengths_.begin() + 1);

    // Make array data for the child type
    type_ = type.value_type();
    type.value_type()->Accept(this);
    auto child_data = *out_data_;

    // Make array data for the list TODO null count == 0
    auto list_type = std::make_shared<arrow::ListType>(type.value_type());
    *out_data_ = arrow::ArrayData::Make(list_type, list_length,
                                        std::move(list_bufs), {child_data}, 0);

    return arrow::Status::OK();
  }

private:
    std::shared_ptr<arrow::DataType> type_;
    std::vector<std::shared_ptr<arrow::Buffer>> buffers_;
    std::vector<int64> lengths_;
    std::shared_ptr<arrow::ArrayData>* out_data_;
};

Status MakeArrayData(std::shared_ptr<arrow::DataType> type,
                     std::vector<int64> array_lengths,
                     std::vector<std::shared_ptr<arrow::Buffer>> buffers,
                     std::shared_ptr<arrow::ArrayData>* out_data) {
  ArrowMakeArrayDataImpl visitor;
  return visitor.Make(type, array_lengths, buffers, out_data);
}




// ---------------------------- simonsom -------------------------------

char* chartobin ( unsigned char c )
{
  static char bin[CHAR_BIT + 1] = { 0 };
  int i;

  for ( i = CHAR_BIT - 1; i >= 0; i-- )
  {
    bin[i] = (c % 2) + '0';
    c /= 2;
  }

  return bin;
}

std::string binaryToString(size_t size, const char* ptr)
{
  unsigned char *b = (unsigned char*) ptr;
  unsigned char byte;
  int i, j;
  std::string res = "";
  std::string current_line = "";
  int counter = 0;
  for (i = 0; i < size; i++) {
    byte = b[i];
    if(counter % 4 == 0) {
      res += current_line;
      current_line = std::string(chartobin(byte)) + "\n";
    } else {
      current_line = std::string(chartobin(byte)) + "\t" + current_line;
    }
    counter = (counter + 1) % 4;
  }
  res += current_line;
  return res;
}




class ConvertToArrowArrayImpl : public arrow::TypeVisitor {

public:
    arrow::Status Make(std::shared_ptr<arrow::DataType> type, std::vector<const char *>& data_column,
                       std::vector<int>& dim_size, std::shared_ptr<arrow::Array>* out_array) {
      type_ = type;
      data_column_ = data_column;
      dims_ = dim_size.size();
      dim_size_ = dim_size;
      out_array_ = out_array;
      empty_shape_ = dims_ == 0;

      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - Make - initialized values:"
                 "\nType: " << type->ToString() << "\ndims_: " << dims_ << "\n";

      ARROW_RETURN_NOT_OK(type->Accept(this));

      return arrow::Status::OK();
    }
protected:
    // helper function to get the size of the current dimension, indexed by builder_idx (reversed to dim)
    size_t getDimSize(int builder_idx) {
      if(empty_shape_) {
        return 1;
      }
      // -1 --> data builder --> dim_size_[dims_ - 1]
      // dims_ - 2 --> outermost tensor builder --> dim_size_[0]
      return dim_size_[dims_ - (builder_idx + 2)];
    }

    template <typename DataTypeType>
    arrow::Status fillData(int data_idx, int& data_offset, std::vector<std::shared_ptr<arrow::ListBuilder>>& builders,
                           std::shared_ptr<arrow::NumericBuilder<DataTypeType>>& data_builder, int current_builder_idx) {


      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - fillData - invoked. Params:\n"
                 "data_idx " << data_idx << "\ndata_offset: " << data_offset << "\ncurrent_builder: " << current_builder_idx;

      // TODO: len_ should be a vector with one value for each list_builder
      // TODO: iterate over all data entries in data_column_

      if(current_builder_idx == -1) {
        using value_type = typename DataTypeType::c_type;
        value_type *data_batch = (value_type *) &(data_column_[data_idx][data_offset]);
        value_type a = data_batch[0];
        VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - fillData - "
                   "\nData batch binary contents (length= " << getDimSize(current_builder_idx) * sizeof(value_type) << ": \n"
                   "" << binaryToString(getDimSize(current_builder_idx) * sizeof(value_type), (char *)data_batch);

        ARROW_RETURN_NOT_OK(data_builder->AppendValues(data_batch, getDimSize(current_builder_idx)));
        data_offset += getDimSize(current_builder_idx) * sizeof(value_type);
        return arrow::Status::OK();
      }

      std::shared_ptr<arrow::ListBuilder> current_builder = builders[current_builder_idx];
      for(int i = 0; i < getDimSize(current_builder_idx); i++) {
        ARROW_RETURN_NOT_OK(current_builder->Append());
        ARROW_RETURN_NOT_OK(fillData<DataTypeType>(data_idx, data_offset, builders,
                data_builder, current_builder_idx-1));
      }

      return arrow::Status::OK();
    }

    template <typename DataTypeType>
    arrow::Status getNestedArray() {
      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - invoked";

      arrow::MemoryPool* pool = arrow::default_memory_pool();

      // data builder for underlying primitive data type of tensor
      std::shared_ptr<arrow::NumericBuilder<DataTypeType>> data_builder =
              std::make_shared<arrow::NumericBuilder<DataTypeType>>(pool);

      // list of builders, one for each additional dimension (d-1) and one outermost
      // builder delimiting the data of each individual tensor
      std::vector<std::shared_ptr<arrow::ListBuilder>> builders;

      // this means that all tensors only hold scalar values (no dimension)
      // this is a special case where we don't delimit the individual tensors with an
      // additional array level.
      if(empty_shape_) {  // TODO: test implementation
        VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - empty_shape_ called";

        for(int i = 0; i < data_column_.size(); i++) {
          int data_offset = 0; // current data offset at data_column[data_idx]
          RETURN_NOT_OK(fillData(i, data_offset, builders, data_builder, -1));
        }

        // finalize and return the array containing all tensors of the column
        std::shared_ptr<arrow::Array> arrow_array;
        RETURN_NOT_OK(data_builder->Finish(&arrow_array));
        *out_array_ = arrow_array;

        return arrow::Status::OK();
      }

      std::shared_ptr<arrow::ListBuilder> b0 =
              std::make_shared<arrow::ListBuilder>(pool, data_builder);
      builders.push_back(b0);
      for(int i = 0; i < dims_ - 1; i++) {
        std::shared_ptr<arrow::ListBuilder> prev = builders.back();
        std::shared_ptr<arrow::ListBuilder> b = std::make_shared<arrow::ListBuilder>(pool, prev);
        builders.push_back(b);
      }

      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - building pools finished";

      // go over all accumulated vectors and build the respective sub-arrays
      for(int i = 0; i < data_column_.size(); i++) {
        builders[dims_ - 1]->Append(); // here starts data of a new tensor

        // this value is passed by ref to share it inside recursive calls to fillData
        int data_offset = 0; // current data offset at data_column[data_idx]
        // feed data to data builder and build shape with list builders corresponding to tensor
        // if dims_ - 2 is negative, we only use the data_builder (no additional nestedness).
        RETURN_NOT_OK(fillData<DataTypeType>(
                i, data_offset, builders, data_builder, dims_ - 2));
      }

      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - successfully read data";


      // finalize and return the array containing all tensors of the column
      std::shared_ptr<arrow::Array> arrow_array;
      RETURN_NOT_OK(builders[dims_-1]->Finish(&arrow_array));

      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - successfully finished array. Size: " << arrow_array->ToString();


      *out_array_ = arrow_array;

      VLOG(0) << "ArrowUtil - ConvertToArrowArrayImpl - NestedArray - written array to out pointer";

      return arrow::Status::OK();
    }

#define VISIT_PRIMITIVE(TYPE)                         \
virtual arrow::Status Visit(const TYPE& type) override {   \
  return getNestedArray<TYPE>();                    \
}

//    VISIT_PRIMITIVE(BooleanType)
    VISIT_PRIMITIVE(arrow::Int8Type)
    VISIT_PRIMITIVE(arrow::Int16Type)
    VISIT_PRIMITIVE(arrow::Int32Type)
    VISIT_PRIMITIVE(arrow::Int64Type)
    VISIT_PRIMITIVE(arrow::UInt8Type)
    VISIT_PRIMITIVE(arrow::UInt16Type)
    VISIT_PRIMITIVE(arrow::UInt32Type)
    VISIT_PRIMITIVE(arrow::UInt64Type)
    VISIT_PRIMITIVE(arrow::HalfFloatType)
    VISIT_PRIMITIVE(arrow::FloatType)
    VISIT_PRIMITIVE(arrow::DoubleType)

private:
    bool empty_shape_;
    std::shared_ptr<arrow::DataType> type_;
    std::vector<const char *> data_column_;
    int dims_;
    std::vector<int> dim_size_;
    std::shared_ptr<arrow::Array>* out_array_;
};

Status GetArrayFromData(std::shared_ptr<arrow::DataType> type, std::vector<const char *>& data_column,
                        std::vector<int>& dim_size, std::shared_ptr<arrow::Array>* out_array) {
  ConvertToArrowArrayImpl visitor;
  CHECK_ARROW(visitor.Make(type, data_column, dim_size, out_array));
}


}  // namespace ArrowUtil
}  // easl
}  // namespace data
}  // namespace tensorflow