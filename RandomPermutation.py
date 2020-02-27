import torch
import numpy as np


class RandomPermutation(object):
    """
    Shuffles point randomly.
    """

    def __call__(self, data):
        np_pos = data.pos.numpy()
        permutation = np.random.permutation(np_pos)
        data.pos = torch.tensor(permutation)
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

