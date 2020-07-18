import torch
from torch_geometric.data import Batch
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes as ps

from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder
from full_network.point_cloud_ae import PointCloudAE
from full_network.middlelayer_encoder import MiddleLayerEncoder, MiddleLayerEncoderSplit
from full_network.middlelayer_decoder import MiddleLayerDecoder, MiddleLayerDecoderSplit


# PATH VARIABLES
RESULT_PATH = "D:/Documenten/Results/Structured/FullAutoEncoder/"
NAME = "LeakyRelu1/"
PATH = RESULT_PATH + NAME

# DATASET VARIABLES
DATASET_SIZE = 1
NB_POINTS = 3600
NORMALS = False

# EPOCH + TESTS
NB_EPOCHS = 65
NB_TESTS = 20

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


def plot_training(network, data_obj, name):
    pos = data_obj.pos
    points_list, batch_list, points_list_out, batch_list_out = network(
        Batch(pos=pos, batch=torch.tensor(0, dtype=torch.long).repeat(pos.size(0)))
    )

    points_np = points_list[0].detach().numpy()
    points_out_np = points_list_out[-1].detach().numpy()

    plot = mp.subplot(
        points_np, c=points_np[:,0],
        s=[2, 1, 0], shading={"point_size": 0.3}
    )
    mp.subplot(
        points_out_np, c=points_out_np[:,0], data=plot,
        s=[2, 1, 1], shading={"point_size": 0.3}
    )

    plot.save(PATH + "visualization_" + name)


def plot_training_layered(network, data_obj, name):
    pos = data_obj.pos
    points_list, batch_list, points_list_out, batch_list_out = network(
        Batch(pos=pos, batch=torch.tensor(0, dtype=torch.long).repeat(pos.size(0)))
    )

    plot = None
    for i in range(NB_LAYERS):
        points_np = points_list[i].detach().numpy()
        points_out_np = points_list_out[i].detach().numpy()
        if plot is None:
            plot = mp.subplot(
                points_np, c=points_np[:, 0],
                s=[2*NB_LAYERS, 1, i], shading={"point_size": 0.3}
            )
        else:
            mp.subplot(
                points_np, c=points_np[:, 0], data=plot,
                s=[2*NB_LAYERS, 1, i], shading={"point_size": 0.3}
            )
        mp.subplot(
            points_out_np, c=points_out_np[:, 0], data=plot,
            s=[2 * NB_LAYERS, 1, NB_LAYERS + i], shading={"point_size": 0.3}
        )

    plot.save(PATH + "visualization_" + name + "_layered")


print("SCRIPT STARTED")
mp.offline()

print("CREATING DATASETS")
print("spheres")
sphere = ps.generate_spheres_dataset(DATASET_SIZE, NB_POINTS, NORMALS)[0]
print("cubes")
cube = ps.generate_cubes_dataset(DATASET_SIZE, NB_POINTS, NORMALS)[0]
print("cylinders")
cylinder = ps.generate_cylinders_dataset(DATASET_SIZE, NB_POINTS, NORMALS)[0]
print("pyramids")
pyramid = ps.generate_pyramids_dataset(DATASET_SIZE, NB_POINTS, NORMALS)[0]
print("torus")
torus = ps.generate_tori_dataset(DATASET_SIZE, NB_POINTS, NORMALS)[0]

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

print("PLOT TRAINING")
print("sphere")
plot_training(net, sphere, "sphere")
plot_training_layered(net, sphere, "sphere")
print("cube")
plot_training(net, cube, "cube")
plot_training_layered(net, cube, "cube")
print("cylinder")
plot_training(net, cylinder, "cylinder")
plot_training_layered(net, cylinder, "cylinder")
print("pyramid")
plot_training(net, pyramid, "pyramid")
plot_training_layered(net, pyramid, "pyramid")
print("torus")
plot_training(net, torus, "torus")
plot_training_layered(net, torus, "torus")







