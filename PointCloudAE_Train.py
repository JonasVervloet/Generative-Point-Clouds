import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ShapeNet
from DatasetFunctionality import ShapeNetFunctionality as snf
from PointCloudAE import PointCloudAE, PointCoudAERandom

import numpy as np
import matplotlib.pyplot as plt

NUMBER_EPOCHS = 50
BATCH_SIZE = 32
RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"

dataset = ShapeNet(root="./data/ShapeNet", categories=["Airplane"])
print("length dataset: " + str(len(dataset)))
print("Total number of nodes: {}".format(snf.get_total_nb_nodes(dataset)))

data = dataset[0]
print(data)

# snf.histogram(dataset, "airplanes")
filtered = snf.filter_data(dataset, 2300)
resampled = snf.simple_resample_to_minimum(filtered)
# resampled = resampled[:1000]
print("len filtered dataset: {}".format(len(filtered)))
print("len resampled dataset: {}".format(len(resampled)))
print("size of data in resampled dataset: {}".format(resampled[5].pos.size()))

train, val = snf.random_split_ratio(resampled, 0.8)
print("length train set: {}".format(len(train)))
print("length val set: {}".format(len(val)))

train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val, batch_size=BATCH_SIZE)


"""NETWORK TRAINING PRESETTING"""

print("cuda available: {}".format(torch.cuda.is_available()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = MSELoss(reduction='mean')


def make_train_step(model, loss_fn, optimizer):
    def train_step(x):
        optimizer.zero_grad()
        model.train()
        out = net(x.pos, x.batch)
        loss = loss_fn(x.pos, out)
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

def make_eval_step(model, loss_fn):
    def eval_step(x):
        model.eval()
        out = net(x.pos, x.batch)
        loss = loss_fn(x.pos, out)
        return loss.item()
    return eval_step


"""RANDOM POOLING AND UNPOOLING NETWORK TRAINING"""

name = "pointcloudae_random"
net = PointCoudAERandom()

optimizer = Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
train_step = make_train_step(net, loss_fn, optimizer)
eval_step = make_eval_step(net, loss_fn)
losses = []
val_losses = []

for epoch in range(NUMBER_EPOCHS):
    print("epoch: {}".format(epoch))
    temp_loss = []
    for batch in train_loader:
        loss = train_step(batch)
        temp_loss.append(loss)
        print(loss)
    losses.append(
        sum(temp_loss) / len(temp_loss)
    )
    print("train loss: {}".format(losses[epoch]))

    temp_loss = []
    with torch.no_grad():
        for batch in val_loader:
            val_loss = eval_step(batch)
            temp_loss.append(val_loss)
        val_losses.append(
            sum(temp_loss) / len(temp_loss)
        )
        print("val loss: {}".format(val_losses[epoch]))
    print("")

torch.save(net.state_dict(), RESULT_PATH + name + "_model.pt")

x = range(len(losses))
plt.plot(x, losses, x, val_losses)
plt.legend(['train loss', 'val loss'])
plt.title('Point Cloud VAE train loss')

plt.savefig(RESULT_PATH + name + '_loss.png')
plt.show()


"""FPS POOLING AND KNN UNPOOLING NETWORK TRAINING"""

name = "pointcloudeae_fps"
net = PointCloudAE()

optimizer = Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
train_step = make_train_step(net, loss_fn, optimizer)
eval_step = make_eval_step(net, loss_fn)
losses = []
val_losses = []

for epoch in range(NUMBER_EPOCHS):
    print("epoch: {}".format(epoch))
    temp_loss = []
    for batch in train_loader:
        loss = train_step(batch)
        temp_loss.append(loss)
        print(loss)
    losses.append(
        sum(temp_loss) / len(temp_loss)
    )
    print("train loss: {}".format(losses[epoch]))

    temp_loss = []
    with torch.no_grad():
        for batch in val_loader:
            val_loss = eval_step(batch)
            temp_loss.append(val_loss)
        val_losses.append(
            sum(temp_loss) / len(temp_loss)
        )
        print("val loss: {}".format(val_losses[epoch]))
    print("")

torch.save(net.state_dict(), RESULT_PATH + name + "_model.pt")

x = range(len(losses))
plt.plot(x, losses, x, val_losses)
plt.legend(['train loss', 'val loss'])
plt.title('Point Cloud VAE train loss')

plt.savefig(RESULT_PATH + name + '_loss.png')
plt.show()
