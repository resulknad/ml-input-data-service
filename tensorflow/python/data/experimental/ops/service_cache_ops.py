from __future__ import absolute_import


import multiprocessing

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


class _ServiceCachePutDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset that allows transparent store to disk."""

    def __init__(self, input_dataset, path, cache_format, cache_compression, parallelism):

        self._input_dataset = input_dataset
        self._path = path
        self._cache_format = cache_format
        self._cache_compression = cache_compression
        self._parallelism = parallelism

        variant_tensor = ged_ops.service_cache_put_dataset(
            input_dataset._variant_tensor,  # pylint: disable=protected-access
            path,
            cache_format,
            cache_compression,
            parallelism,
            **self._flat_structure)
        super(_ServiceCachePutDataset, self).__init__(input_dataset, variant_tensor)


    def _transformation_name(self):
        return "Dataset.serviceCachePutDataset"


@tf_export("data.experimental.service_cache_put")
def service_cache_put(path, cache_format=2, cache_compression=1, parallelism=8):

    def _apply_fn(dataset):
        """Actual dataset transformation."""
        project_func = None
        dataset = _ServiceCachePutDataset(
            input_dataset=dataset,
            path=path,
            cache_format=cache_format,
            cache_compression=cache_compression,
            parallelism=parallelism)

        return dataset

    return _apply_fn

class _ServiceCacheGetDataset(dataset_ops.DatasetSource):
  """A dataset that gets data from the tf.data service cache. (for testing
  only)"""

  def __init__(self, path, cache_format, cache_compression, parallelism, element_spec):

    self._path = path
    self._cache_format = cache_format
    self._cache_compression = cache_compression
    self._parallelism = parallelism
    self._element_spec = element_spec

    variant_tensor = ged_ops.service_cache_get_dataset(
      path, cache_format, cache_compression, parallelism, **self._flat_structure)

    super(_ServiceCacheGetDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec

@tf_export("data.experimental.serviceCacheGetDataset")
def service_cache_get(
  path,
  cache_format=2,
  cache_compression=1,
  parallelism=8,
  element_spec=tensor_spec.TensorSpec(shape=(),dtype=dtypes.int32)):
  return _ServiceCacheGetDataset(path, cache_format, cache_compression, parallelism, element_spec)



class _MarkerDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset that allows to add a marker node in the graph representation."""

    def __init__(self, input_dataset, marker_type):

        self._input_dataset = input_dataset
        self._marker_type = marker_type

        variant_tensor = ged_ops.marker_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            marker_type=self._marker_type,
            **self._flat_structure)

        super(_MarkerDataset, self).__init__(input_dataset, variant_tensor)


    def _transformation_name(self):
        return "Dataset.markerDataset"


@tf_export("data.experimental.mark")
def service_cache_put(marker_type="source_cache"):

    def _apply_fn(dataset):
        """Actual dataset transformation."""
        project_func = None
        dataset = _MarkerDataset(
            input_dataset=dataset,
            marker_type=marker_type)

        return dataset

    return _apply_fn
