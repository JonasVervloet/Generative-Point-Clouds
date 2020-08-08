import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from dataset.primitives import PrimitiveShapes
from LossFunctions import ChamferDistLoss, ChamferVAELoss
from full_network.full_network_vae import FullNetworkVAE

FROM_EPOCH = 450
NB_EPOCHS = 550
BATCH_SIZE = 5
RESULT_PATH = "D:/Resultaten/Reverse/VAE/"
NAME = "FullNetwork/"
START_LR = 0.00001
original_lr = 0.001
LR_NB = 1
NB_POINTS = 3600
TRAIN_SIZE = 50
VAL_SIZE = 5

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
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
print(len(train_loader))
print(len(val_loader))

loss_fn = ChamferVAELoss()

for lr in range(LR_NB):
    learning_rate = START_LR / max(1, (lr * 10))
    path = RESULT_PATH + NAME + "LearningRate{}/".format(round(original_lr * 100000))
    print(learning_rate)
    print(round(learning_rate*100000))

    net = FullNetworkVAE()
    train_losses = np.empty(0)
    val_losses = np.empty(0)
    if not FROM_EPOCH == 0:
        print('loaded net')
        net.load_state_dict(
            torch.load(path + "model_epoch{}.pt".format(FROM_EPOCH))
        )
        train_losses = np.load(path + "trainloss_epoch{}.npy".format(FROM_EPOCH))
        val_losses = np.load(path + "valloss_epoch{}.npy".format(FROM_EPOCH))
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    net.train()
    for i in range(NB_EPOCHS + 1 - FROM_EPOCH):
        epoch = i + FROM_EPOCH
        print(epoch)
        temp_loss = []

        if FROM_EPOCH != 0 and epoch == FROM_EPOCH:
            print('continue')
            continue

        for batch in train_loader:
            optimizer.zero_grad()

            pos = batch.pos
            batch_inds = batch.batch

            points_out, batch_out, mean, log_variance = net(pos, batch_inds)
            loss = loss_fn(
                input=pos,
                output=points_out,
                batch_in=batch_inds,
                batch_out=batch_out,
                mean=mean,
                log_variance=log_variance
            )
            loss.backward()
            optimizer.step()
            temp_loss.append(loss.item())

        train_loss = sum(temp_loss) / len(train_loader)
        print(train_loss)
        train_losses = np.append(train_losses, train_loss)

        net.eval()
        temp_loss = []
        for val_batch in val_loader:
            pos = val_batch.pos
            batch_inds = val_batch.batch

            points_out, batch_out, mean, log_variance = net(pos, batch_inds)
            loss = loss_fn(
                input=pos,
                output=points_out,
                batch_in=batch_inds,
                batch_out=batch_out,
                mean=mean,
                log_variance=log_variance
            )
            temp_loss.append(loss.item())

        val_loss = sum(temp_loss) / len(val_loader)
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

