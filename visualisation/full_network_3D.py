import torch
from torch_geometric.data import DataLoader
import meshplot as mp
import numpy as np

from dataset.primitive_shapes import PrimitiveShapes as ps
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder
from full_network.point_cloud_ae import PointCloudAE
from full_network.middlelayer_encoder import MiddleLayerEncoder
from full_network.middlelayer_decoder import MiddleLayerDecoder


# PATH VARIABLES
RESULT_PATH = "D:/Documenten/Results/Structured/FullAutoEncoder/"
NAME = "LeakyRelu1/"
PATH = RESULT_PATH + NAME

# DATASET VARIABLES
DATASET_SIZE = 10
NB_POINTS = 3600
NORMALS = False
BATCH_SIZE = 5

# EPOCH
NB_EPOCHS = 65

# FULL AUTOENCODER NETWORK VARIABLES
NB_LAYERS = 3
NBS_NEIGHS = [25, 16, 9]
RADII = [0.23, 1.3, 2.0]
LEAKY = True


def get_neighborhood_encoder(latent_size, mean):
    return NeighborhoodEncoder(
        nbs_features=[32, 64, 64],
        nbs_features_global=[64, 32, latent_size],
        mean=mean,
        leaky=LEAKY
    )


def get_neighborhood_decoder(latent_size, nb_neighbors):
    return GridDeformationDecoder(
        input_size=latent_size,
        nbs_features_global=[32, 64, 64],
        nbs_features=[64, 32, 3],
        nb_neighbors=nb_neighbors,
        leaky=LEAKY
    )


# ENCODERS AND DECODERS
LAT1 = 8
LAT2 = 64
LAT3 = 128
MEAN = False

neigh_enc1 = get_neighborhood_encoder(LAT1, MEAN)
encoder1 = neigh_enc1
neigh_enc2 = get_neighborhood_encoder(LAT1, MEAN)
encoder2 = MiddleLayerEncoder(
    neighborhood_enc=neigh_enc2,
    input_size=LAT1,
    nbs_features=[64, 128, 128],
    nbs_features_global=[128, 64, LAT2],
    mean=MEAN,
    leaky=LEAKY
)
neigh_enc3 = get_neighborhood_encoder(LAT1, MEAN)
encoder3 = MiddleLayerEncoder(
    neighborhood_enc=neigh_enc3,
    input_size=LAT2,
    nbs_features=[128, 256, 256],
    nbs_features_global=[256, 128, LAT3],
    mean=MEAN,
    leaky=LEAKY
)

neigh_dec1 = get_neighborhood_decoder(LAT1, NBS_NEIGHS[-1])
decoder1 = MiddleLayerDecoder(
    neighborhood_dec=neigh_dec1,
    input_size=LAT3,
    nbs_features_global=[128, 256, LAT1],
    nbs_features=[128, 256, LAT2],
    leaky=LEAKY
)
neigh_dec2 = get_neighborhood_decoder(LAT1, NBS_NEIGHS[-2])
decoder2 = MiddleLayerDecoder(
    neighborhood_dec=neigh_dec2,
    input_size=LAT2,
    nbs_features_global=[64, 128, LAT1],
    nbs_features=[64, 128, LAT1],
    leaky=LEAKY
)
neigh_dec3 = get_neighborhood_decoder(LAT1, NBS_NEIGHS[-3])
decoder3 = neigh_dec3

ENCODERS = [encoder1, encoder2, encoder3]
DECODERS = [decoder1, decoder2, decoder3]


def evaluate_dataset(network, dataset):
    network.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    latent = None
    for batch in loader:
        encoded, points_list, batch_list = network.encode(batch.pos, batch.batch)
        if latent is None:
            latent = encoded
        else:
            latent = torch.cat([latent, encoded], dim=0)

    return latent


print("SCRIPT STARTED")
mp.offline()

print("CREATING DATASETS")
print("spheres")
spheres = ps.generate_spheres_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("cubes")
cubes = ps.generate_cubes_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("cylinders")
cylinders = ps.generate_cylinders_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("pyramids")
pyramids = ps.generate_pyramids_dataset(DATASET_SIZE, NB_POINTS, NORMALS)
print("torus")
tori = ps.generate_tori_dataset(DATASET_SIZE, NB_POINTS, NORMALS)

print("LOADING NETWORK")
net = PointCloudAE(
    nb_layers=NB_LAYERS,
    nbs_neighbors=NBS_NEIGHS,
    radii=RADII,
    encoders=ENCODERS,
    decoders=DECODERS
)
net.load_state_dict(
        torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(NB_EPOCHS))
    )
net.eval()

print("ENCODING")
print("spheres")
spheres_encoded = evaluate_dataset(net, spheres)
print("cubes")
cubes_encoded = evaluate_dataset(net, cubes)
print("cylinders")
cylinders_encoded = evaluate_dataset(net, cylinders)
print("pyramids")
pyramids_encoded = evaluate_dataset(net, pyramids)
print("torus")
tori_encoded = evaluate_dataset(net, tori)

encoded_points = torch.cat(
    [spheres_encoded,
     cubes_encoded,
     cylinders_encoded,
     pyramids_encoded,
     tori_encoded],
    dim=0
).detach().numpy()
print(encoded_points.shape)
print(np.amax(encoded_points, axis=0))
print(np.amin(encoded_points, axis=0))
# encoded_points = encoded_points/np.amax(encoded_points, axis=0)[0]

# sphere_color = np.tile(
#     np.array([1.0, 0.0, 0.0]),
#     (spheres_encoded.shape[0], 1)
# )
# cube_color = np.tile(
#     np.array([0.0, 1.0, 0.0]),
#     (cubes_encoded.shape[0], 1)
# )
# cylinder_color = np.tile(
#     np.array([0.0, 0.0, 1.0]),
#     (cylinders_encoded.shape[0], 1)
# )
# pyramid_color = np.tile(
#     np.array([1.0, 1.0, 0.0]),
#     (pyramids_encoded.shape[0], 1)
# )
# torus_color = np.tile(
#     np.array([0.0, 1.0, 1.0]),
#     (tori_encoded.shape[0], 1)
# )
#
# colors = np.concatenate(
#     [sphere_color,
#      cube_color,
#      cylinder_color,
#      pyramid_color,
#      torus_color],
#     axis=0
# )
# print(colors.shape)
#
# plot = mp.plot(
#     encoded_points, c=colors, filename=PATH + "3D_plot"
# )

