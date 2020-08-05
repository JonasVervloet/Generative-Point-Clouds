import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

<<<<<<< HEAD
RESULT_PATH = "D:/Documenten/Results/"

<<<<<<< HEAD
arr1 = np.array(
    [[1, 2, 3, 4, 5, 6]]
)
arr2 = np.array(
    [4, 5, 6, 7, 8, 9]
)
arr2 = [arr2]

concat = np.concatenate((arr1, arr2), axis=0)
print(concat)
=======
test1 = [
    [1, 2, 3, 4, 5],
]
arr1 = np.array(test1)
test2 = [
    [16, 17, 18, 19, 20]
]
arr2 = np.array(test2)
test3 = [
    [16, 17, 18, 19, 20]
]
arr3 = np.array(test3)

res1 = np.concatenate((arr1, arr2), axis=0)
res2 = np.concatenate((res1, arr3), axis=0)

print(res1)
print(res2)


>>>>>>> parent of 5fafec9... visualization interpolation
=======
from dataset.primitives import PrimitiveShapes as ps
import meshplot as mp

from relative_layer.decoder3 import RelativeDecoder2
from relative_layer.decoder import SimpleRelativeDecoder
from relative_layer.rotation_invariant_layer import RotationInvariantLayer
from relative_layer.encoder3 import RotationInvariantEncoder

RESULT_PATH = "D:/Documenten/Results/"

<<<<<<< HEAD
points = torch.rand([7200, 3])
batch = torch.arange(2).repeat_interleave(3600)
batch_obj = Batch(batch=batch, pos=points)

network = SingleLayerNetwork(1, [25], False, radius=0.3)
encoder = NeighborhoodEncoder(
    nb_features=[80, 40, 20], mean=False
)
decoder = NeighborhoodDecoder(
    nb_features=[20, 40, 80], nb_neighbors=25
)
network.set_encoder(encoder)
network.set_decoder(decoder)
inp, batch_in, outp, batch_out = network(batch_obj)
print(inp.size())
print(batch_in.size())
print(outp.size())
print(batch_out.size())
>>>>>>> parent of 7b76629... Pre cleaning

=======
test = torch.tensor([1, 2, 3, 4, 5])
test2 = test.clone()
test2[0] = 5
print(test)
print(test2)
>>>>>>> parent of 0fe5a94... Single Layer Network












