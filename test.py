import torch
from torch_geometric.data import Batch
from dataset.primitive_shapes import PrimitiveShapes as ps
from network_text_converter import NetworkTextFileConverter
from network_manager import NetworkManager
from loss_function import ChamferDistLoss, ChamferVAELoss, LayerChamferDistLoss, ChamferDistLossFullNetwork
from full_network.network_generator import NetworkGenerator

from full_network.middlelayer_decoder import MiddleLayerDecoder, MiddleLayerDecoderSplit
from full_network.middlelayer_encoder import MiddleLayerEncoder, MiddleLayerEncoderSplit
from full_network.point_cloud_ae import PointCloudAE
from relative_layer.grid_deform_decoder import GridDeformationDecoder
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder

COMPLETE_ENCODING = False

print()
print("Test Generator")
print()

generator = NetworkGenerator()
generator.set_nb_layers(2)

generator.add_nb_neighbor(25)
generator.add_nb_neighbor(16)
# generator.add_nb_neighbor(9)

generator.add_radius(0.23)
generator.add_radius(1.3)
# generator.add_radius(2.0)

generator.add_latent_size(8)
generator.add_latent_size(64)
# generator.add_latent_size(128)

generator.deactivate_leaky_relu()
if COMPLETE_ENCODING:
    generator.make_complete_encoding()
else:
    generator.make_incomplete_encoding()
generator.set_max_pooling()
generator.set_grid_decoder()
generator.disable_split_networks()

network = generator.generate_network()
print(network.to_string())
print()

test_points = torch.randn(7200, 3)
test_batch = torch.arange(2).repeat_interleave(3600)

spheres = ps.generate_spheres_dataset(2, 3600, normals=False)
spheres_input = torch.cat([spheres[0].pos, spheres[1].pos], 0)
batch = Batch(pos=spheres_input, batch=test_batch)
print("input: ")
print(spheres_input.size())
print(test_batch.size())

input_points, input_clusters, features, fps_points = network.encode(
    batch.pos, batch.batch
)

print()
print("decoding")
print()

relative_points_list, clusters_list = network.decode_features(features)
assert(len(relative_points_list) == len(clusters_list))
for i in range(len(relative_points_list)):
    print(relative_points_list[i].size())
    print(clusters_list[i].size())

print()

output_points = network.construct_output_points(relative_points_list, clusters_list, fps_points)
for i in range(len(output_points)):
    print(output_points[i].size())

print()

output_clusters = network.construct_output_clusters(clusters_list, features)
for i in range(len(output_clusters)):
    print(output_clusters[i].size())

print()
output_points, output_clusters = network.decode(features, fps_points)
assert(len(output_points) == len(output_clusters))
for i in range(len(output_points)):
    print(output_points[i].size())
    print(output_clusters[i].size())

