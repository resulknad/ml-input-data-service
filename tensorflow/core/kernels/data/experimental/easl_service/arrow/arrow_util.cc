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

#include "tensorflow/core/kernels/data/experimental/easl_service/arrow/arrow_util.h"

#include <utility>
#include "arrow/adapters/tensorflow/convert.h"
#include "arrow/api.h"
#include "arrow/io/file.h"
#include "arrow/ipc/feather.h"
#include "arrow/ipc/api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace easl {
namespace ArrowUtil {

ArrowMetadata::ArrowMetadata() {
  this->num_worker_threads_ = 0;
}

Status ArrowMetadata::WriteData(const std::string& path) {

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;

  std::shared_ptr<arrow::Array> rowShapeArr;
  arrow::StringBuilder rowShapeBuilder(pool);

  VLOG(0) << "Writing metadata to directory [" << path << "].";
  VLOG(0) << "Number of tensor shapes: [" << shapes_.size() << "].";
  VLOG(0) << "Partial batches: [" << partial_batching_ << "].";

  for(TensorShape &s : shapes_) {
    TensorShapeProto p;
    s.AsProto(&p);
    std::string s_str = p.SerializeAsString();
    rowShapeBuilder.Append(s_str.data(), s_str.length());
  }
  rowShapeBuilder.Finish(&rowShapeArr);
  arrays.push_back(rowShapeArr);

  // a bit of a hack: encode experimental bool in header of first column
  std::string header = experimental_ ? "experimental" : "standard";
  schema_vector.push_back(arrow::field(header, rowShapeArr->type()));



  auto itr = partial_batch_shapes_.begin();
  for(auto i : partial_batch_shapes_) {
    std::string col_name(i.first.data(), i.first.length());

    std::shared_ptr<arrow::Array> partialShapesArr;
    arrow::StringBuilder partialShapesBuilder(pool);

    for(TensorShape &s : i.second) {
      TensorShapeProto p;
      s.AsProto(&p);
      std::string s_str = p.SerializeAsString();
      partialShapesBuilder.Append(s_str.data(), s_str.length());
    }

    partialShapesBuilder.Finish(&partialShapesArr);
    arrays.push_back(partialShapesArr);
    schema_vector.push_back(arrow::field(col_name, partialShapesArr->type()));
  }


  // create schema
  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, arrays);

  // open file
  std::shared_ptr<arrow::io::FileOutputStream> file;
  file = arrow::io::FileOutputStream::Open(
          io::JoinPath(path, "arrow_metadata.feather"), /*append=*/false).MoveValueUnsafe();

  // write table to file
  struct arrow::ipc::feather::WriteProperties wp = {
          arrow::ipc::feather::kFeatherV2Version,
          // Number of rows per intra-file chunk. Use smaller chunksize when you need faster random row access
          1LL << 16,
          arrow::Compression::UNCOMPRESSED,
          arrow::util::kUseDefaultCompressionLevel
  };
  arrow::ipc::feather::WriteTable(*table, file.get(), wp);

  return Status::OK();
}

int ArrowMetadata::WriteMetadataToFile(const std::string& path) {
  mutex_lock l(mu_);  // unlocked automatically upon function return
  --num_worker_threads_;
  if(num_worker_threads_ == 0) {
    WriteData(path);
    return 1;
  }

  return 0;
}

Status ArrowMetadata::ReadMetadataFromFile(Env* env, const std::string& path) {

  // check whether file exists, if not -> assume standard format
  Status s = env->FileExists(io::JoinPath(path, "arrow_metadata.feather"));
  if (s != Status::OK()) {
    experimental_ = false;
    partial_batching_ = false;

    return Status::OK();
  }

  // clear metadata if already data in it
  this->partial_batch_shapes_.clear();
  this->shapes_.clear();

  // open file
  std::shared_ptr<arrow::Table> table;
  std::shared_ptr<arrow::io::MemoryMappedFile> mm_file;
  mm_file = arrow::io::MemoryMappedFile::Open(io::JoinPath(path, "arrow_metadata.feather"),
          arrow::io::FileMode::READ).MoveValueUnsafe();
  std::shared_ptr<arrow::ipc::feather::Reader> reader;
  reader = arrow::ipc::feather::Reader::Open(mm_file).MoveValueUnsafe();
  reader->Read(&table);

  arrow::TableBatchReader tr(*table.get());
  std::shared_ptr<arrow::RecordBatch> batch;
  std::shared_ptr<arrow::RecordBatch> next_batch;
  tr.ReadNext(&batch); // should never be nullptr
  tr.ReadNext(&next_batch);
  if(batch == nullptr) {
    return Status(error::UNAVAILABLE, "Metadata empty.");
  } else if(next_batch != nullptr) {  // next batch should be nullptr
    return Status(error::UNAVAILABLE, "Metadata too large");
  }

  // read whether experimental:
  this->experimental_ = batch->column_name(0) == "experimental";

  // read rowShape (column 0 always exists)
  arrow::StringArray* rowShapeArr = dynamic_cast<arrow::StringArray*>(batch->column(0).get());
  for(int i = 0; i < rowShapeArr->length(); i++) {
    std::string proto_data = rowShapeArr->GetString(i);
    TensorShapeProto s_proto;
    s_proto.ParseFromString(proto_data);
    TensorShape s(s_proto);
    shapes_.push_back(s);
  }

  // read partialTensorShapes
  partial_batching_ = batch->num_columns() > 1;
  for(int i = 1; i < batch->num_columns(); i++) {
    std::string col_name = batch->column_name(i);
    std::vector<TensorShape> partial_shapes;
    arrow::StringArray* partialShapeArr = dynamic_cast<arrow::StringArray*>(batch->column(i).get());

    for(int j = 0; j < partialShapeArr->length(); j++) {
      std::string proto_data = partialShapeArr->GetString(j);
      TensorShapeProto s_proto;
      s_proto.ParseFromString(proto_data);
      TensorShape s(s_proto);
      partial_shapes.push_back(s);
    }
    partial_batch_shapes_.insert({col_name, partial_shapes});
  }

  return Status::OK();
}

Status ArrowMetadata::AddPartialBatch(const string& doc, const std::vector<TensorShape>& last_batch_shape) {
  mutex_lock l(mu_);  // unlocked automatically upon function return
  this->partial_batch_shapes_.insert({doc, last_batch_shape});
  this->partial_batching_ = true;

  return Status::OK();
}

Status ArrowMetadata::GetPartialBatches(string doc, std::vector<TensorShape> *out_last_batch_shape) {
  auto it = partial_batch_shapes_.find(doc);
  if (it != partial_batch_shapes_.end()) {
    *out_last_batch_shape = partial_batch_shapes_[doc];

  } else {  // no partial batches for this file
    // do nothing as out_last_batch_shape points to empty vector already
  }
  return Status::OK();
}

bool ArrowMetadata::IsPartialBatching() {
  return partial_batching_;
}

Status ArrowMetadata::GetRowShape(std::vector<TensorShape> *out_row_shape) {
  *out_row_shape = shapes_;

  return Status::OK();
}

Status ArrowMetadata::GetRowDType(std::vector<DataType> *out_row_dtype) {
  *out_row_dtype = dtypes_;
  return Status::OK();
}

Status ArrowMetadata::SetRowDType(std::vector<DataType> row_dtypes) {
  if(dtypes_.empty()) {
    this->dtypes_ = std::move(row_dtypes);
  }
  return Status::OK();
}

Status ArrowMetadata::SetRowShape(std::vector<TensorShape> row_shape) {  //TODO: probably don't need lock here
  if(shapes_.empty()) {
    mutex_lock l(mu_);  // unlocked automatically upon function return
    this->shapes_ = std::move(row_shape);
  }

  return Status::OK();
}

Status ArrowMetadata::RegisterWorker() {
  mutex_lock l(mu_);  // unlocked automatically upon function return
  num_worker_threads_++;
  return Status::OK();
}

bool ArrowMetadata::IsExperimental() {
  return experimental_;
}

Status ArrowMetadata::SetExperimental(bool exp) {
  this->experimental_ = exp;
  return Status::OK();
}

Status ArrowMetadata::AddLastRowBatch(Tensor &t) {
  last_row_batches_.emplace_back(std::move(t));
  return Status::OK();
}

Status ArrowMetadata::GetLastRowBatch(std::vector<Tensor> *out) {
  mutex_lock l(mu_);  // unlocked automatically upon function return
  --num_worker_threads_;
  if(num_worker_threads_ == 0) {
    *out = last_row_batches_;
  }
  return Status::OK();
}



// -----------------------------------------------------------------------------
// Conversion Arrow - Tensorflow
// -----------------------------------------------------------------------------



Status GetTensorFlowType(const std::shared_ptr<::arrow::DataType>& dtype,
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


      i_ = i;  // we want to get i-th element of array -> tensor data
      batch_size_ = batch_size;
      out_shape_ = out_shape;  //out_shape_points to shape needed for allocation
      out_dtype_ = out_dtype;

      // batch_size of 0 indicates 1 record at a time, no batching
      if (batch_size_ > 0) {
        out_shape_->AddDim(batch_size_);
      }

      CHECK_ARROW(array->Accept(this));  // visit all types of arrow arrays
      return Status::OK();
    }

protected:
    template <typename ArrayType>
    arrow::Status VisitPrimitive(const ArrayType& array) {

      if (out_dtype_ != nullptr) {
        GetTensorFlowType(array.type(), out_dtype_);
        return ::arrow::Status::OK();
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

      // values in the outermost array are tensors with their corresponding dimensionality and can
      // be indexed by i_. if we flatten first dimension, value_offset(i_) gives the new idx.
      int32 values_offset = array.value_offset(i_);
       int32 array_length = array.value_length(i_); // #elements in values array belonging to i_

      // Add dimension for array
      // --> first time, this is going to be null --> only want to add dim for elements of array
      if (out_shape_ != nullptr) {
        out_shape_->AddDim(array_length);
      }

      // Prepare the array data buffer and visit the array slice
      // this function returns an array where the outermost dimension is flattened
      std::shared_ptr<arrow::Array> values = array.values();
      std::shared_ptr<arrow::Array> element_values =
              values->Slice(values_offset, array_length);

      // for subsequent dimensions always look at first element
      i_ = 0;
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


      const auto& fw_type =
              static_cast<const arrow::FixedWidthType&>(*array.type());
      const int64_t type_width = fw_type.bit_width() / 8;

      // Primitive Arrow arrays have validity and value buffers, currently
      // only arrays with null count == 0 are supported, so only need values here
      static const int VALUE_BUFFER = 1;
      // same as array.values()
      auto values = array.data()->buffers[VALUE_BUFFER];  // only works for primitive arrays!
      if (values == NULLPTR) {
        return arrow::Status::Invalid(
                "Received an Arrow array with a NULL value buffer");
      }

      const void* src =
              (values->data() + array.data()->offset * type_width) + i_ * type_width;   // i_ is 0 if from list_array
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

      int32 values_offset = array.value_offset(i_);   // i_ is always 0 except for the outermost call
      int32 curr_array_length = array.value_length(i_);

      // Save current index and swap after array is copied
      int32 tmp_index = i_;
      i_ = 0;

      // Prepare the array data buffer and visit the array slice
      std::shared_ptr<arrow::Array> values = array.values();
      std::shared_ptr<arrow::Array> element_values =
              values->Slice(values_offset, curr_array_length);
      auto result = element_values->Accept(this);

      // Reset state variables for next time
      i_ = tmp_index;
      return result;
    }

    virtual arrow::Status Visit(const arrow::StringArray& array) override {

      int32 curr_array_length = array.length();

      tstring* strings = reinterpret_cast<tstring*>(out_tensor_->data()); // TODO: check if this works!

      if(out_tensor_->NumElements() > 1) {  // not a scalar tensor -> array only holds elements of this tensor
        for(int i = 0; i< curr_array_length; i++) {
          strings[i] = tstring(array.GetString(i).data());
        }
      } else {  // for scalar tensors array also holds elements of other tensors --> use i_
        strings[0] = tstring(array.GetString(i_).data());
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
                       const absl::InlinedVector<int64, 4>& dim_size, std::shared_ptr<arrow::Array>* out_array,
                       const absl::InlinedVector<int64, 4>& last_dim_size) {
      data_column_ = data_column;
      dims_ = dim_size.size();
      dim_size_ = &dim_size;
      last_dim_size_ = &last_dim_size;
      out_array_ = out_array;
      empty_shape_ = dims_ == 0;
      pool_ = arrow::default_memory_pool();

      ARROW_RETURN_NOT_OK(type->Accept(this));

      return arrow::Status::OK();
    }
protected:
// helper function to get the size of the current dimension, indexed by builder_idx (reversed to dim)
    size_t getDimSize(int builder_idx, int data_idx) {
      if(empty_shape_) {
        return 1;
      }

      if(data_idx == data_column_.size() - 1) { // last row -> last_dim_size
        return (*last_dim_size_)[dims_ - (builder_idx + 2)];
      }
      // -1 --> data builder --> dim_size_[dims_ - 1]
      // dims_ - 2 --> outermost tensor builder --> dim_size_[0]
      return (*dim_size_)[dims_ - (builder_idx + 2)];
    }

    template <typename c_type, typename builder_type>
    arrow::Status fillData(int data_idx, int& data_offset, std::vector<std::shared_ptr<arrow::ListBuilder>>& builders,
                           std::shared_ptr<builder_type>& data_builder, int current_builder_idx) {

      // base case of recursion. if current_builder_idx == -1 we use the data_builder to actually insert data.
      if(current_builder_idx == -1) {
        c_type *data_batch = (c_type *) (data_column_[data_idx] + (data_offset));

        ARROW_RETURN_NOT_OK(data_builder->AppendValues(data_batch, getDimSize(current_builder_idx, data_idx)));
        data_offset += getDimSize(current_builder_idx, data_idx) * sizeof(c_type);
        return arrow::Status::OK();
      }

      // we use the append function to delimit beginning of new subarray.
      std::shared_ptr<arrow::ListBuilder> current_builder = builders[current_builder_idx];
      for(int i = 0; i < getDimSize(current_builder_idx, data_idx); i++) {
        ARROW_RETURN_NOT_OK(current_builder->Append());
        ARROW_RETURN_NOT_OK( (fillData<c_type, builder_type>(data_idx, data_offset, builders,
                data_builder, current_builder_idx-1)) );
      }

      return arrow::Status::OK();
    }

    template <typename c_type, typename builder_type>
    arrow::Status getNestedArray(std::shared_ptr<builder_type>& data_builder) {

      // list of builders, one for each additional dimension (d-1) and one outermost
      // builder delimiting the data of each individual tensor
      std::vector<std::shared_ptr<arrow::ListBuilder>> builders;

      // this means that all tensors only hold scalar values (no dimension)
      // this is a special case where we don't delimit the individual tensors with an
      // additional array level.
      if(empty_shape_) {
        for(int i = 0; i < data_column_.size(); i++) {
          int data_offset = 0; // current data offset at data_column[data_idx]
          ARROW_RETURN_NOT_OK( (fillData<c_type, builder_type>(i, data_offset, builders, data_builder, -1)) );
        }

        // finalize and return the array containing all tensors of the column
        std::shared_ptr<arrow::Array> arrow_array;
        ARROW_RETURN_NOT_OK(data_builder->Finish(&arrow_array));
        *out_array_ = arrow_array;
        return arrow::Status::OK();
      }

      // construct builder dependencies
      std::shared_ptr<arrow::ListBuilder> b0 =
              std::make_shared<arrow::ListBuilder>(pool_, data_builder);
      builders.push_back(b0);
      for(int i = 0; i < dims_ - 1; i++) {
        std::shared_ptr<arrow::ListBuilder> prev = builders.back();
        std::shared_ptr<arrow::ListBuilder> b = std::make_shared<arrow::ListBuilder>(pool_, prev);
        builders.push_back(b);
      }


      // go over all accumulated vectors and build the respective sub-arrays
      for(int i = 0; i < data_column_.size(); i++) {
        ARROW_RETURN_NOT_OK(builders[dims_ - 1]->Append()); // here starts data of a new tensor

        // this value is passed by ref to share it inside recursive calls to fillData
        int data_offset = 0; // current data offset at data_column[data_idx]
        // feed data to data builder and build shape with list builders corresponding to tensor
        // if dims_ - 2 is negative, we only use the data_builder (no additional nestedness).
        ARROW_RETURN_NOT_OK( (fillData<c_type, builder_type>(
                i, data_offset, builders, data_builder, dims_ - 2)) );
      }


      // finalize and return the array containing all tensors of the column
      std::shared_ptr<arrow::Array> arrow_array;
      ARROW_RETURN_NOT_OK(builders[dims_-1]->Finish(&arrow_array));


      *out_array_ = arrow_array;

      return arrow::Status::OK();
    }

#define VISIT_PRIMITIVE(TYPE)                                           \
virtual arrow::Status Visit(const TYPE& type) override {                \
using c_type = typename TYPE::c_type;                                   \
using builder_type = arrow::NumericBuilder<TYPE>;                       \
std::shared_ptr<builder_type> n =                                       \
              std::make_shared<builder_type>(pool_);                    \
return getNestedArray<c_type, builder_type>(n);                         \
}

// Visit numerical values
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

    virtual arrow::Status Visit(const arrow::StringType& type) override {
      using c_type = const char *;
      using builder_type = arrow::StringBuilder;
      std::shared_ptr<builder_type> s = std::make_shared<builder_type>(pool_);
      return getNestedArray<c_type, builder_type>(s);
    }

    virtual arrow::Status Visit(const arrow::BooleanType& type) override {
      using c_type = const uint8_t;
      using builder_type = arrow::BooleanBuilder;
      std::shared_ptr<builder_type> b = std::make_shared<builder_type>(pool_);
      return getNestedArray<c_type, builder_type>(b);
    }

private:
    bool empty_shape_;

// for numerical data
    std::vector<const char *> data_column_;

    arrow::MemoryPool* pool_;
    int dims_;
    const absl::InlinedVector<int64, 4> *dim_size_;
    const absl::InlinedVector<int64, 4> *last_dim_size_;
    std::shared_ptr<arrow::Array>* out_array_;
};

arrow::Status GetArrayFromData(std::shared_ptr<arrow::DataType> type, std::vector<const char *>& data_column,
                               const absl::InlinedVector<int64, 4>& dim_size, std::shared_ptr<arrow::Array>* out_array,
                               const absl::InlinedVector<int64, 4>& last_dim_size) {
  ConvertToArrowArrayImpl visitor; // use visitor pattern to cover all types
  return visitor.Make(type, data_column, dim_size, out_array, last_dim_size);
}


// ***************************** Experimental ********************************


// currently not supported: strings
arrow::Status GetArrayFromDataExperimental(
        size_t buff_len,
        std::vector<const char *>& data_column,
        std::shared_ptr<arrow::Array>* out_array,
        int64 last_buff_len) {
  arrow::StringBuilder data_builder(arrow::default_memory_pool());


  for(int i = 0; i < data_column.size() - 1; i++) {
    const char* buff = data_column[i];
    ARROW_RETURN_NOT_OK(data_builder.Append(buff, buff_len)) ;
  }
  // last tensor possibly partially batched data
  const char* buff = data_column[data_column.size() - 1];


  std::shared_ptr<arrow::Array> arrow_array;
  ARROW_RETURN_NOT_OK(data_builder.Finish(&arrow_array));

  *out_array = arrow_array;

  return arrow::Status::OK();
}

Status AssignTensorExperimental(
        std::shared_ptr<arrow::Array> array,
        int64 i,
        Tensor* out_tensor) {

  arrow::StringArray* str_arr = dynamic_cast<arrow::StringArray*>(array.get());

  int64 value_offset = str_arr->value_offset(i);
  size_t len = str_arr->value_offset(i+1) - value_offset;  // Note: no out of bounds error

  const void* src = str_arr->raw_data() + value_offset;

  void* dst = const_cast<char*>(out_tensor->tensor_data().data());
  memcpy(dst, src, len);

  return Status::OK();
}


}  // namespace ArrowUtil
}  // easl
}  // namespace data
}  // namespace tensorflow
