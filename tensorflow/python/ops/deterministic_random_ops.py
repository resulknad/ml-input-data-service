"""Deterministic random ops which do not take a seed as a tensor input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# import collections
# import contextlib
# import copy
# import os
# import random
# import threading
# from absl import logging
# import six

from tensorflow.python.compat import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
# Work with the context and seed set in the starting
from tensorflow.python.framework import random_seed
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
#from tensorflow.python.ops import gen_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Helper to be used to update the context with num element seeds
@tf_export("random.experimental.stateless_split")
@dispatch.add_dispatch_support
def split(seed, num):
  """Splits an RNG seed into `num` new seeds by adding a leading axis.
  Example:
  >>> seed = [1, 2]
  >>> new_seeds = tf.random.experimental.stateless_split(seed, num=3)
  >>> print(new_seeds)
  tf.Tensor(
  [[1105988140 1738052849]
   [-335576002  370444179]
   [  10670227 -246211131]], shape=(3, 2), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.59835213, -0.9578608 ,
  0.9002807 ], dtype=float32)>
  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or
      `int64`). (When using XLA, only `int32` is allowed.)
    num: optional, a positive integer or scalar tensor indicating the number of
      seeds to produce (default 2).
  Returns:
    A tensor with shape [num, 2] representing `num` new seeds. It will have the
    same dtype as `seed` (if `seed` doesn't have an explict dtype, the dtype
    will be determined by `tf.convert_to_tensor`).
  """
  seed = ops.convert_to_tensor(seed)
  return deterministic_random_uniform(shape=[num, 2], seed=seed, dtype=seed.dtype,
                                  minval=None, maxval=None)


@tf_export("random.experimental.stateless_fold_in")
@dispatch.add_dispatch_support
def fold_in(seed, data):
  """Folds in data to an RNG seed to form a new RNG seed.
  For example, in a distributed-training setting, suppose we have a master seed
  and a replica ID. We want to fold the replica ID into the master seed to
  form a "replica seed" to be used by that replica later on, so that different
  replicas will generate different random numbers but the reproducibility of the
  whole system can still be controlled by the master seed:
  >>> master_seed = [1, 2]
  >>> replica_id = 3
  >>> replica_seed = tf.random.experimental.stateless_fold_in(
  ...   master_seed, replica_id)
  >>> print(replica_seed)
  tf.Tensor([1105988140          3], shape=(2,), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=replica_seed)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.03197195, 0.8979765 ,
  0.13253039], dtype=float32)>
  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or
      `int64`). (When using XLA, only `int32` is allowed.)
    data: an `int32` or `int64` scalar representing data to be folded in to the
      seed.
  Returns:
    A new RNG seed that is a deterministic function of the inputs and is
    statistically safe for producing a stream of new pseudo-random values. It
    will have the same dtype as `data` (if `data` doesn't have an explict dtype,
    the dtype will be determined by `tf.convert_to_tensor`).
  """
  data = ops.convert_to_tensor(data)
  seed1 = stateless_random_uniform(shape=[], seed=seed, dtype=data.dtype,
                                   minval=None, maxval=None)
  return array_ops.stack([seed1, data])


def _get_key_counter_alg(seed):
  if compat.forward_compatible(2021, 3, 1):
    key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed)
    alg = gen_stateless_random_ops_v2.stateless_random_get_alg()
    return key, counter, alg
  else:
    return gen_stateless_random_ops_v2.stateless_random_get_key_counter_alg(
        seed)


@tf_export("random.stateless_uniform")
@dispatch.add_dispatch_support
def deterministic_random_uniform(shape,
                             seed,
                             minval=0,
                             maxval=None,
                             dtype=dtypes.float32,
                             name=None):
  """Outputs deterministic pseudorandom values from a uniform distribution.
  This is a stateless version of `tf.random.uniform`: if run twice with the
  same seeds and shapes, it will produce the same pseudorandom numbers.  The
  output is consistent across multiple runs on the same hardware (and between
  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.
  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.
  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.
  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).
  For full-range (i.e. inclusive of both max and min) random integers, pass
  `minval=None` and `maxval=None` with an integer `dtype`. For an integer dtype
  either both `minval` and `maxval` must be `None` or neither may be `None`. For
  example:
  ```python
  ints = tf.random.stateless_uniform(
      [10], seed=(2, 3), minval=None, maxval=None, dtype=tf.int32)
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    minval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The lower bound on the range of random values to
      generate. Pass `None` for full-range integers.  Defaults to 0.
    maxval: A Tensor or Python value of type `dtype`, broadcastable with
      `shape` (for integer types, broadcasting is not supported, so it needs to
      be a scalar). The upper bound on the range of random values to generate.
      Defaults to 1 if `dtype` is floating point. Pass `None` for full-range
      integers.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32`, or
      `int64`. For unbounded uniform ints (`minval`, `maxval` both `None`),
      `uint32` and `uint64` may be used.
    name: A name for the operation (optional).
  Returns:
    A tensor of the specified shape filled with random uniform values.
  Raises:
    ValueError: If `dtype` is integral and only one of `minval` or `maxval` is
      specified.
  """
  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                   dtypes.float64, dtypes.int32, dtypes.int64, dtypes.uint32,
                   dtypes.uint64):
    raise ValueError("Invalid dtype %r" % dtype)
  if dtype.is_integer:
    if (minval is None) != (maxval is None):
      raise ValueError("For integer dtype {}, minval and maxval must be both "
                       "`None` or both non-`None`.".format(dtype))
    if minval is not None and dtype in (dtypes.uint32, dtypes.uint64):
      raise ValueError("Invalid dtype for bounded uniform integers: %r" % dtype)
  elif maxval is None:
    maxval = 1
  with ops.name_scope(name, "stateless_random_uniform",
                      [shape, seed, minval, maxval]) as name:
    shape = tensor_util.shape_tensor(shape)
    if dtype.is_integer and minval is None:
      if compat.forward_compatible(2020, 10, 25):
        key, counter, alg = _get_key_counter_alg(seed)
        result = (gen_stateless_random_ops_v2
                  .stateless_random_uniform_full_int_v2(
                      shape, key=key, counter=counter, dtype=dtype, alg=alg,
                      name=name))
      else:
        result = gen_stateless_random_ops.stateless_random_uniform_full_int(
            shape, seed=seed, dtype=dtype, name=name)
    else:
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
      if dtype.is_integer:
        if compat.forward_compatible(2020, 10, 25):
          key, counter, alg = _get_key_counter_alg(seed)
          result = gen_stateless_random_ops_v2.stateless_random_uniform_int_v2(
              shape, key=key, counter=counter, minval=minval, maxval=maxval,
              alg=alg, name=name)
        else:
          result = gen_stateless_random_ops.stateless_random_uniform_int(
              shape, seed=seed, minval=minval, maxval=maxval, name=name)
      else:
        if compat.forward_compatible(2020, 10, 25):
          key, counter, alg = _get_key_counter_alg(seed)
          rnd = gen_stateless_random_ops_v2.stateless_random_uniform_v2(
              shape, key=key, counter=counter, dtype=dtype, alg=alg)
        else:
          rnd = gen_stateless_random_ops.stateless_random_uniform(
              shape, seed=seed, dtype=dtype)
        result = math_ops.add(rnd * (maxval - minval), minval, name=name)
    tensor_util.maybe_set_static_shape(result, shape)
    return result


@tf_export("random.stateless_normal")
@dispatch.add_dispatch_support
def stateless_random_normal(shape,
                            seed,
                            mean=0.0,
                            stddev=1.0,
                            dtype=dtypes.float32,
                            name=None):
  """Outputs deterministic pseudorandom values from a normal distribution.
  This is a stateless version of `tf.random.normal`: if run twice with the
  same seeds and shapes, it will produce the same pseudorandom numbers.  The
  output is consistent across multiple runs on the same hardware (and between
  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    name: A name for the operation (optional).
  Returns:
    A tensor of the specified shape filled with random normal values.
  """
  with ops.name_scope(name, "stateless_random_normal",
                      [shape, seed, mean, stddev]) as name:
    shape = tensor_util.shape_tensor(shape)
    mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
    stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
    if compat.forward_compatible(2021, 3, 1):
      key, counter, alg = _get_key_counter_alg(seed)
      rnd = gen_stateless_random_ops_v2.stateless_random_normal_v2(
          shape, key=key, counter=counter, dtype=dtype, alg=alg)
    else:
      rnd = gen_stateless_random_ops.stateless_random_normal(shape, seed, dtype)
    result = math_ops.add(rnd * stddev, mean, name=name)
    tensor_util.maybe_set_static_shape(result, shape)
    return result