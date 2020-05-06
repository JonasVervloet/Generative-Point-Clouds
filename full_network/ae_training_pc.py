import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from dataset.primitive_shapes import PrimitiveShapes
from loss_function import ChamferDistLoss
from loss_function import LayerChamferDistLoss

from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder
from full_network.middlelayer_encoder import MiddleLayerEncoder
from full_network.middlelayer_decoder import MiddleLayerDecoder
from full_network.point_cloud_ae import PointCloudAE

#  PATH VARIABLES
RESULT_PATH = "D:/Documenten/Documenten Molenpolder/Jonas/Results/FullNetwork/"
NAME = "LeakyRelu1/"
PATH = RESULT_PATH + NAME

# EPOCH + LEARNING RATE
FROM_EPOCH = 0
END_EPOCH = 100
LEARNING_RATE = 0.001

# DATASET VARIABLES
NB_POINTS = 3600
SPHERE = True
CUBE = True
CYLINDER = True
PYRAMID = True
TORUS = True
SHAPES = [SPHERE, CUBE, CYLINDER, PYRAMID, TORUS]
NORMALS = False
TRAIN_SIZE = 100
VAL_SIZE = 10
BATCH_SIZE = 5

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
LAT1 = 4
LAT2 = 8
LAT3 = 3
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

print("SCRIPT STARTED")
print("DATASET PREP")
print("train data")
train_dataset = PrimitiveShapes.generate_dataset(
    nb_objects=TRAIN_SIZE, nb_points=NB_POINTS,
    shapes=SHAPES, normals=NORMALS
)
print("validation data")
val_dataset = PrimitiveShapes.generate_dataset(
    nb_objects=VAL_SIZE, nb_points=NB_POINTS,
    shapes=SHAPES, normals=NORMALS
)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("train loader length: {}".format(len(train_loader)))
print("val loader length: {}".format(len(val_loader)))

loss_fn = ChamferDistLoss()

print("NETWORK SETUP")
net = PointCloudAE(
    nb_layers=NB_LAYERS,
    nbs_neighbors=NBS_NEIGHS,
    radii=RADII,
    encoders=ENCODERS,
    decoders=DECODERS
)

train_losses = np.empty(0)
val_losses = np.empty(0)

if not FROM_EPOCH == 0:
    print('loading net...')
    net.load_state_dict(
        torch.load(PATH + "model_epoch{}.pt".format(FROM_EPOCH))
    )
    train_losses = np.load(PATH + "trainloss_epoch{}.npy".format(FROM_EPOCH))
    val_losses = np.load(PATH + "valloss_epoch{}.npy".format(FROM_EPOCH))

optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

print("TRAINING STARTING")
print("learning rate: {}".format(LEARNING_RATE))
print("from epoch: {}".format(FROM_EPOCH))
print("end epoch: {}".format(END_EPOCH))
for i in range(END_EPOCH + 1 - FROM_EPOCH):
    epoch = i + FROM_EPOCH
    print("epoch {}".format(epoch))

    if not FROM_EPOCH == 0 and epoch == FROM_EPOCH:
        continue

    net.train()
    temp_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        points_list, batch_list, points_list_out, batch_list_out = net(batch)
        points_in = points_list[0]
        batch_in = batch_list[0]
        points_out = points_list_out[-1]
        batch_out = batch_list_out[-1]

        loss = loss_fn(
            points_in,
            points_out,
            batch_in=batch_in,
            batch_out=batch_out
        )
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())

    train_loss = sum(temp_loss) / (TRAIN_SIZE * sum(SHAPES))
    train_losses = np.append(train_losses, train_loss)
    print("train loss: {}".format(train_loss))

    net.eval()
    temp_loss = []
    for val_batch in val_loader:
        points_list, batch_list, points_list_out, batch_list_out = net(batch)
        points_in = points_list[0]
        batch_in = batch_list[0]
        points_out = points_list_out[-1]
        batch_out = batch_list_out[-1]

        loss = loss_fn(
            points_in,
            points_out,
            batch_in=batch_in,
            batch_out=batch_out
        )
        temp_loss.append(loss.item())

    val_loss = sum(temp_loss) / (VAL_SIZE * sum(SHAPES))
    val_losses = np.append(val_losses, val_loss)
    print("val loss: {}".format(val_loss))

    if epoch % 5 == 0:
        np.save(PATH + "trainloss_epoch{}.npy".format(epoch), train_losses)
        np.save(PATH + "valloss_epoch{}.npy".format(epoch), val_losses)
        torch.save(
            net.state_dict(),
            PATH + "model_epoch{}.pt".format(epoch)
        )
        plt.clf()
        x = range(len(train_losses))
        plt.plot(x, train_losses, x, val_losses)
        plt.legend(['train loss', 'validation loss'])
        plt.title('Point AutoEncoder Network train loss')
        plt.yscale('log')
        plt.savefig(
            PATH + "loss_epoch{}.png".format(epoch)
        )