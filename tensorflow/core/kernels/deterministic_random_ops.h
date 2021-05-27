// Taken and modified from core/kernels/stateless_random_ops_v2.h

#ifndef TENSORFLOW_CORE_KERNELS_DETERMINISTIC_RANDOM_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DETERMINISTIC_RANDOM_OPS_H_

#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

inline Status CheckKeyCounterShape(Algorithm const& alg,
                                   TensorShape const& key_shape,
                                   TensorShape const& counter_shape) {
  if (!(key_shape.dims() == 1 && key_shape.dim_size(0) == RNG_KEY_SIZE)) {
    return errors::InvalidArgument(
        "key must have shape [", RNG_KEY_SIZE, "], not ",
        key_shape.DebugString(),
        ". (Note that batched keys are not supported yet.)");
  }
  auto counter_size = GetCounterSize(alg);
  if (!(counter_shape.dims() == 1 &&
        counter_shape.dim_size(0) >= counter_size)) {
    return errors::InvalidArgument(
        "counter must be a vector with length at least ", counter_size,
        "; got shape: ", counter_shape.DebugString(),
        ". (Note that batched counters are not supported yet.)");
  }
  return Status::OK();
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DETERMINISTIC_RANDOM_OPS_H_