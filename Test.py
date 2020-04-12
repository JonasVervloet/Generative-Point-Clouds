import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

from dataset.primitives import PrimitiveShapes as ps
import meshplot as mp
from torch_geometric.data import Batch

from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder

RESULT_PATH = "D:/Documenten/Results/"

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













