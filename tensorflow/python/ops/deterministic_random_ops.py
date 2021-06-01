"""Deterministic random ops which do not take a seed as a tensor input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import collections
import contextlib
import copy
import os
import random
import threading
from absl import logging
import six

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
#from tensorflow.python.ops import gen_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

def stateless_random_uniform(shape,
                             minval=0,
                             maxval=None,
                             dtype=dtypes.float32,
                             name=None):

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
