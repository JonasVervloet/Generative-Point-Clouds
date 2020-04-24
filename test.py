import torch
import torch.nn as nn
import numpy as np
import math
import point_cloud_utils as pcu

from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder

from full_network.middlelayer_encoder import MiddleLayerEncoder
from full_network.middlelayer_decoder import MiddleLayerDecoder
from full_network.point_cloud_ae import PointCloudAE

from torch_geometric.data import Batch

RESULT_PATH = "D:/Documenten/Results/"

print(torch.cuda.is_available())

decoder = GridDeformationDecoder(
    input_size=32,
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
    nb_neighbors=25
)
print(decoder)

latent_vec = torch.rand((5, 32))
print(latent_vec.size())

decoded, cluster = decoder(latent_vec)
print(decoded.size())
print(cluster.size())















