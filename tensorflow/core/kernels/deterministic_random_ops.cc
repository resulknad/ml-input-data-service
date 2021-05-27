// Taken from core/kernels/stateless_random_ops_v2.cc
// Currently the OP registered for this purpose exploits the seed to generate a key and counter
// Which is further used by the kernel's computation for the purpose of random numbers generation.
// Using the Global_seed thus involves adding functionalities like split and key and counter generation
// at python/ops/deterministic_random_ops.py


#include "tensorflow/core/kernels/deterministic_random_ops.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/random_poisson_op.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return Status::OK();
}

class DeterministicRandomOpBase : public OpKernel {
 public:
  explicit DeterministicRandomOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Sanitize input
    const Tensor& shape_t = ctx->input(0);
    const Tensor& key_t = ctx->input(1);
    const Tensor& counter_t = ctx->input(2);
    const int alg_input_idx = 3;
    const Tensor& alg_t = ctx->input(alg_input_idx);

    int alg_id;
    OP_REQUIRES_OK(ctx, GetScalar(alg_t, alg_input_idx, &alg_id));
    Algorithm alg = Algorithm(alg_id);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    OP_REQUIRES_OK(ctx,
                   CheckKeyCounterShape(alg, key_t.shape(), counter_t.shape()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) {
      return;
    }

    // Fill in the random numbers
    Fill(ctx, alg, key_t, counter_t, output);
  }

  // The part of Compute that depends on device, type, and distribution
  virtual void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                    const Tensor& counter, Tensor* output) = 0;
};

template <typename Device, typename Distribution>
class DeterministicRandomOp : public DeterministicRandomOpBase {
 public:
  using DeterministicRandomOpBase::DeterministicRandomOpBase;

  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
    typedef typename Distribution::ResultElementType T;
    auto flat = output->flat<T>();
    if (alg == RNG_ALG_PHILOX) {
      // Reuse the compute kernels from the stateful random ops
      auto key_data = key.flat<uint64>().data();
      auto counter_data = counter.flat<uint64>().data();
      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), key_data, counter_data,
          random::PhiloxRandom() /*dummy*/, flat.data(), flat.size(),
          Distribution());
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

