import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

from dataset.primitives import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer
from LossFunctions import ChamferDistLoss
<<<<<<< HEAD
from relative_layer.single_layer_network import SingleLayerNetwork
from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder

# PATH VARIABLES
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> parent of 7d59a97... Training file update
RESULT_PATH_LAPTOP = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
RESULT_PATH_PC = "D:/Documenten/Documenten Molenpolder/Jonas/Results/"
LAPTOP = False
if LAPTOP:
    RESULT_PATH = RESULT_PATH_LAPTOP
else:
    RESULT_PATH = RESULT_PATH_PC
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> parent of 9835611... update
=======
>>>>>>> parent of 7d59a97... Training file update
NAME = "ParameterReduction4/"
=======
=======
RESULT_PATH = "D:/Documenten/Results/Structured/SingleLayerNetwork/"
>>>>>>> parent of 4a0c8b0... PC result path added
NAME = "ParameterReduction2/"
>>>>>>> parent of 47fc858... parameter reduction 3
=======
NAME = "Network5/"
>>>>>>> parent of 1ecc13d... Radii + experiments excel added
=======
NAME = "Network3/"
>>>>>>> parent of 7b76629... Pre cleaning
PATH = RESULT_PATH + NAME
=======
>>>>>>> parent of f8bf2d5... Update training file SingleLayerNetwork

import matplotlib.pyplot as plt

FROM_EPOCH = 0
NB_EPOCHS = 200
RESULT_PATH = "D:/Documenten/Results/"
NAME = "LearningRateMean/"
START_LR = 0.0001
LR_NB = 1
<<<<<<< HEAD
NB_NEIGHS = 25
NB_NEIGHS2 = 16
NB_NEIGHS3 = 9
NB_POINTS = 3600
TRAIN_SIZE = 200
VAL_SIZE = 20
RADIUS = 0.23
<<<<<<< HEAD
if FINAL_LAYER:
    NB_CLUSTER = 1
else:
    nb_points = NB_POINTS
    for i in range(NB_LAYERS):
        nb_points /= NBS_NEIGHS[i]
    NB_CLUSTER = nb_points

# ENCODER AND DECODER
ENCODER = NeighborhoodEncoder(
<<<<<<< HEAD
    nbs_features=[32, 64, 64],
<<<<<<< HEAD
<<<<<<< HEAD
    nbs_features_global=[64, 32, 4],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=4,
=======
    nbs_features_global=[64, 32, 16],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=16,
>>>>>>> parent of 47fc858... parameter reduction 3
=======
    nbs_features_global=[64, 32, 32],
    mean=False
)
DECODER = GridDeformationDecoder(
    input_size=32,
>>>>>>> parent of 1ecc13d... Radii + experiments excel added
    nbs_features_global=[32, 64, 64],
    nbs_features=[64, 32, 3],
=======
    nbs_features=[80, 40, 20],
    pool_index=1,
    mean=False
)
DECODER = NeighborhoodDecoder(
    nbs_features=[20, 40, 80],
    unpool_index=2,
>>>>>>> parent of 7b76629... Pre cleaning
    nb_neighbors=NBS_NEIGHS[-1]
)
=======
RADIUS2 = 1.3
RADIUS3 = 2.0
>>>>>>> parent of f8bf2d5... Update training file SingleLayerNetwork
=======
>>>>>>> parent of 377b2d6... UPDATE


print("STARTING TRAINTNG")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(30, 2000)
loader = DataLoader(dataset=dataset, batch_size=1)
print(len(loader))

loss_fn = ChamferDistLoss()

for lr in range(LR_NB):
    learning_rate = START_LR / max(1, (lr * 10))
    path = RESULT_PATH + NAME + "LearningRate{}/".format(round(learning_rate * 100000))
    print(learning_rate)
    print(round(learning_rate*100000))

    net = SimpleRelativeLayer(20, 20, 10, 5, mean=True)
    if not FROM_EPOCH == 0:
        print('loaded net')
        net.load_state_dict(
            torch.load(path + "model_epoch{}.pt".format(FROM_EPOCH))
        )
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    net.train()
    losses = []

    for i in range(NB_EPOCHS - FROM_EPOCH):
        epoch = i + FROM_EPOCH
        print(epoch)
        temp_loss = []

        for batch in loader:
            optimizer.zero_grad()

            origin, cluster, output = net(batch.pos)
            loss = loss_fn(origin, output, batch_in=cluster)
            loss.backward()
            optimizer.step()
            temp_loss.append(loss)

        losses.append(
            sum(temp_loss) / (len(loader) * 20)
        )

        if epoch % 5 == 0:
            torch.save(
                net.state_dict(),
                path + "model_epoch{}.pt".format(epoch)
            )
            plt.clf()
            x = range(FROM_EPOCH, epoch + 1)
            plt.plot(x, losses)
            plt.legend(['train loss'])
            plt.title('Simple Relative Layer train loss')
            plt.yscale('log')
            plt.savefig(
                path + "loss_epoch{}.png".format(epoch)
            )









