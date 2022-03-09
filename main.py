import tensorflow as tf
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops  import UnaryUnchangedStructureDataset, SkipDataset

class SkiponeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` skipping every other element from its input."""

  def __init__(self, input_dataset):
    """See `Dataset.skip()` for details."""
    self._input_dataset = input_dataset
    #print(vars(gen_dataset_ops))
    variant_tensor = gen_dataset_ops.skipone_dataset(
        input_dataset._variant_tensor, **self._flat_structure)
    super(SkiponeDataset, self).__init__(input_dataset, variant_tensor)

dataset = SkiponeDataset(tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8]))

for element in dataset:
  print(element)
