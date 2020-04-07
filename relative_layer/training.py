import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

from dataset.primitives import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer
from relative_layer.simple_layer2 import SimpleRelativeLayer2
from relative_layer.simple_layer3 import SimpleRelativeLayer3
from relative_layer.decoder2 import RelativeDecoder
from relative_layer.encoder2 import RelativeEncoder
from relative_layer.decoder3 import RelativeDecoder2
from relative_layer.rotation_invariant_layer import RotationInvariantLayer
from LossFunctions import ChamferDistLoss

import matplotlib.pyplot as plt
import numpy as np

FROM_EPOCH = 20
NB_EPOCHS = 30
RESULT_PATH = "D:/Documenten/Results/"
NAME = "LearningRate2/"
START_LR = 0.001
LR_NB = 1
NB_NEIGHS = 25
NB_NEIGHS2 = 16
NB_NEIGHS3 = 9
NB_POINTS = 3600
TRAIN_SIZE = 200
VAL_SIZE = 20
RADIUS = 0.23
RADIUS2 = 1.3
RADIUS3 = 2.0


print("STARTING TRAINTNG")
print("DATASET PREP")
print("train data")
train_dataset = PrimitiveShapes.generate_dataset(
    TRAIN_SIZE, NB_POINTS,
    shapes=[True, True, True, True, True], normals=False
)
print("validation data")
val_dataset = PrimitiveShapes.generate_dataset(
    VAL_SIZE, NB_POINTS,
    shapes=[True, True, True, True, True], normals=False
)
train_loader = DataLoader(dataset=train_dataset, batch_size=5)
val_loader = DataLoader(dataset=val_dataset, batch_size=5)
print(len(train_loader))
print(len(val_loader))

loss_fn = ChamferDistLoss()

for lr in range(LR_NB):
    learning_rate = START_LR / max(1, (lr * 10))
    path = RESULT_PATH + NAME + "LearningRate{}/".format(round(learning_rate * 100000))
    print(learning_rate)
    print(round(learning_rate*100000))

    train_losses = np.empty(0)
    val_losses = np.empty(0)
    # net = SimpleRelativeLayer(NB_NEIGHS, 80, 40, 20, radius=RADIUS, mean=False)
    # net.set_decoder(RelativeDecoder2(NB_NEIGHS, 20, 40, 80))
    # net = RotationInvariantLayer(NB_NEIGHS, RADIUS, 80, 40, 20)
    # net = SimpleRelativeLayer2(NB_NEIGHS, NB_NEIGHS2, 80, 40, 20, RADIUS2)
    net = SimpleRelativeLayer3(NB_NEIGHS, NB_NEIGHS2, NB_NEIGHS3, 80, 40, 20, radius=RADIUS3)

    if not FROM_EPOCH == 0:
        print('LOADING NET...')
        net.load_state_dict(
            torch.load(path + "model_epoch{}.pt".format(FROM_EPOCH))
        )
        train_losses = np.load(path + "trainloss_epoch{}.npy".format(FROM_EPOCH))
        val_losses = np.load(path + "valloss_epoch{}.npy".format(FROM_EPOCH))

    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    for i in range(NB_EPOCHS + 1 - FROM_EPOCH):
        epoch = i + FROM_EPOCH
        print(epoch)

        if not FROM_EPOCH == 0 and epoch == FROM_EPOCH:
            continue

        net.train()
        temp_loss = []
        for batch in train_loader:
            optimizer.zero_grad()

            origin, cluster, output = net(batch.pos, batch.batch)
            # origin, cluster, output, cluster_out = net(batch.pos, batch.norm)
            loss = loss_fn(origin, output, batch_in=cluster)
            # loss = loss_fn(origin, output, batch_in=cluster, batch_out=cluster_out)
            loss.backward()
            optimizer.step()
            temp_loss.append(loss)

        train_loss = (sum(temp_loss) / (len(train_loader) * NB_NEIGHS)).detach().numpy()
        print(train_loss)
        train_losses = np.append(train_losses, train_loss)

        net.eval()
        temp_loss = []
        for val_batch in val_loader:
            origin, cluster, output = net(val_batch.pos, batch.batch)
            # origin, cluster, output, cluster_out = net(val_batch.pos, val_batch.norm)
            loss = loss_fn(origin, output, batch_in=cluster)
            # loss = loss_fn(origin, output, batch_in=cluster, batch_out=cluster_out)
            temp_loss.append(loss.item())

        val_loss = sum(temp_loss) / (len(val_loader) * NB_NEIGHS)
        print(val_loss)
        val_losses = np.append(val_losses, val_loss)

        if epoch % 5 == 0:
            np.save(path + "trainloss_epoch{}.npy".format(epoch), train_losses)
            np.save(path + "valloss_epoch{}.npy".format(epoch), val_losses)
            torch.save(
                net.state_dict(),
                path + "model_epoch{}.pt".format(epoch)
            )
            plt.clf()
            x = range(len(train_losses))
            plt.plot(x, train_losses, x, val_losses)
            plt.legend(['train loss', 'validation loss'])
            plt.title('Simple Relative Layer train loss')
            plt.yscale('log')
            plt.savefig(
                path + "loss_epoch{}.png".format(epoch)
            )









