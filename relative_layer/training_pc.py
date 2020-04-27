import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from dataset.primitive_shapes import PrimitiveShapes
from loss_function import ChamferDistLoss
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder

import datetime

begin_time = datetime.datetime.now()
print("Begin-time: {}".format(begin_time))

# PATH VARIABLES
RESULT_PATH_LAPTOP = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
RESULT_PATH_PC = "D:/Documenten/Documenten Molenpolder/Jonas/Results/"
LAPTOP = True
if LAPTOP:
    RESULT_PATH = RESULT_PATH_LAPTOP
else:
    RESULT_PATH = RESULT_PATH_PC
NAME = "ParameterReduction3/"
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

# SINGLE LAYER NETWORK VARIABLES.
CUDA = True
DEVICE = torch.device(
    "cuda:0" if CUDA and torch.cuda.is_available() else "cpu"
)
NB_LAYERS = 1
NBS_NEIGHS = [25]
FINAL_LAYER = False
RADIUS = 0.23
if FINAL_LAYER:
    NB_CLUSTER = 1
else:
    nb_points = NB_POINTS
    for i in range(NB_LAYERS):
        nb_points /= NBS_NEIGHS[i]
    NB_CLUSTER = nb_points

# ENCODER AND DECODER
ENCODER = NeighborhoodEncoder(
    nbs_features=[32, 64, 64],
    nbs_features_global=[64, 32, 8],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=8,
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
    nb_neighbors=NBS_NEIGHS[-1]
)


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
print("nb clusters: {}".format(NB_CLUSTER))
print("nb shape types: {}".format(sum(SHAPES)))
print("nb layers: {}".format(NB_LAYERS))
print("nbs neighbors: {}".format(NBS_NEIGHS))
print("final layer: {}".format(FINAL_LAYER))
print("radius: {}".format(RADIUS))
net = SingleLayerNetwork(
    nb_layers=NB_LAYERS,
    nbs_neighbors=NBS_NEIGHS,
    final_layer=FINAL_LAYER,
    radius=RADIUS,
    device=DEVICE
)
net.set_encoder(ENCODER)
net.set_decoder(DECODER)

train_losses = np.empty(0)
val_losses = np.empty(0)

if not FROM_EPOCH == 0:
    print('loading net...')
    net.load_state_dict(
        torch.load(PATH + "model_epoch{}.pt".format(FROM_EPOCH))
    )
    train_losses = np.load(PATH + "trainloss_epoch{}.npy".format(FROM_EPOCH))
    val_losses = np.load(PATH + "valloss_epoch{}.npy".format(FROM_EPOCH))
net.to(DEVICE)
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
        input_points, cluster_in, output_points, cluster_out = net(batch)
        loss = loss_fn(input_points, output_points, batch_in=cluster_in, batch_out=cluster_out)
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())

    train_loss = sum(temp_loss) / (TRAIN_SIZE * sum(SHAPES) * NB_CLUSTER)
    train_losses = np.append(train_losses, train_loss)
    print("train loss: {}".format(train_loss))

    net.eval()
    temp_loss = []
    for val_batch in val_loader:
        input_points, cluster_in, output_points, cluster_out = net(batch)
        loss = loss_fn(input_points, output_points, batch_in=cluster_in, batch_out=cluster_out)
        temp_loss.append(loss.item())

    val_loss = sum(temp_loss) / (VAL_SIZE * sum(SHAPES) * NB_CLUSTER)
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
        plt.title('Simple Relative Layer train loss')
        plt.yscale('log')
        plt.savefig(
            PATH + "loss_epoch{}.png".format(epoch)
        )

end_time = datetime.datetime.now()
print("End-time: {}".format(end_time))

print("Execution time: {}".format(end_time - begin_time))