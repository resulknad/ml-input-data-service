/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/data/skipone_dataset_op.h"

#include <algorithm>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const SkiponeDatasetOp::kDatasetType;
/* static */ constexpr const char* const SkiponeDatasetOp::kInputDataset;
/*
constexpr const char* const BatchDatasetOp::kBatchSize;
constexpr const char* const BatchDatasetOp::kDropRemainder;
constexpr const char* const BatchDatasetOp::kParallelCopy;
constexpr const char* const BatchDatasetOp::kOutputTypes;
constexpr const char* const BatchDatasetOp::kOutputShapes;
 */

//constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kSkiponeDataset[] = "SkiponeDataset";

class SkiponeDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        op_version_(op_version) {
    input_->Ref();
    }
    // NOTE(mrry): Currently we implement "batch up to" semantics. If
    // we could tell statically that the input dataset is infinite,
    // then we could always report `batch_size` as the 0th dimension.
   /* const auto& input_shapes = input_->output_shapes();
    output_shapes_.reserve(input_shapes.size());
    for (const auto& input_shape : input_shapes) {
      if (drop_remainder_ || input_->Cardinality() == kInfiniteCardinality) {
        output_shapes_.emplace_back(
            PartialTensorShape({batch_size_}).Concatenate(input_shape));
      } else {
        output_shapes_.emplace_back(
            PartialTensorShape({-1}).Concatenate(input_shape));
      }
    }*/


  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
      return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override {
    int64 n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / 2;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      // Each row of `batch_elements` is a tuple of tensors from the
      // input iterator.

        mutex_lock l(mu_);
        if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
        }
        int num_skipped;
        TF_RETURN_IF_ERROR(input_impl_->Skip(ctx, 1,
                                             end_of_sequence, &num_skipped));
        if (*end_of_sequence) {
            // We reached the end before the count was reached.
            input_impl_.reset();
            return Status::OK();
        }
        // Return GetNext() on the underlying iterator.
        TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        return Status::OK();

    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), 1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (!input_impl_) {
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      /*if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }*/
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const DatasetBase* const input_;
  const int op_version_;
  std::vector<PartialTensorShape> output_shapes_;
  const TraceMeMetadata traceme_metadata_;
};

SkiponeDatasetOp::SkiponeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kSkiponeDataset ? 1 : 2) {

}

void SkiponeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {

  *output = new Dataset(ctx, input,
                        op_version_);

}

namespace {
REGISTER_KERNEL_BUILDER(Name("SkiponeDataset").Device(DEVICE_CPU),
                        SkiponeDatasetOp);

REGISTER_KERNEL_BUILDER(Name("SkiponeDatasetV2").Device(DEVICE_CPU),
                        SkiponeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
