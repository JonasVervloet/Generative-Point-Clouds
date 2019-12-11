import torch
from torch.nn import MSELoss, Linear
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ShapeNet
from DatasetFunctionality import ShapeNetFunctionality as snf
from PointCloudAE import PointCloudAE, PointCoudAERandom, SimplePointCloudAE
from torch_geometric.nn import DynamicEdgeConv

import numpy as np
import matplotlib.pyplot as plt

NUMBER_EPOCHS = 50
BATCH_SIZE = 16
NB_VERTICES = 2300
RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"


def batch_loss(input, output, loss_function):
    reshaped_input = snf.reshape_batch(input, NB_VERTICES)
    reshaped_output = snf.reshape_batch(output, NB_VERTICES)
    return loss_function(reshaped_output, reshaped_input)


def make_train_step(model, loss_function, opt):
    def train_step(x):
        model.train()
        opt.zero_grad()
        out = model(x.pos, x.batch)
        loss = batch_loss(x.pos, out, loss_function)
        loss.backward()
        opt.step()
        return loss.item()
    return train_step


def make_eval_step(model, loss_function):
    def eval_step(x):
        model.eval()
        out = model(x.pos, x.batch)
        loss = batch_loss(x.pos, out, loss_function)
        return loss.item()
    return eval_step


def train_network(model, model_name, opt, loss_function):
    train_step = make_train_step(model, loss_function, opt)
    eval_step = make_eval_step(model, loss_function)
    losses = []
    val_losses = []

    for epoch in range(NUMBER_EPOCHS):
        print("epoch: {}".format(epoch))
        print("TRAIN")
        temp_loss = []
        i = 0
        for batch in train_loader:
            i += 1
            print("batch {}".format(i))
            loss = train_step(batch)
            temp_loss.append(loss)
        losses.append(
            sum(temp_loss)/len(temp_loss)
        )
        print("train loss: {}".format(losses[epoch]))

        print("EVAL")
        temp_loss = []
        with torch.no_grad():
            i = 0
            for batch in val_loader:
                i += 1
                print("batch {}".format(i))
                val_loss = eval_step(batch)
                temp_loss.append(val_loss)
            val_losses.append(
                sum(temp_loss)/len(temp_loss)
            )
            print("val loss: {}".format(val_losses[epoch]))
        print("")

        torch.save(
            net.state_dict(), RESULT_PATH + model_name + "/" + "model_batchsize{}_epoch{}.pt".format(BATCH_SIZE, epoch)
        )
        plt.clf()
        x = range(len(losses))
        plt.plot(x, losses, x, val_losses)
        plt.legend(['train loss', 'val loss'])
        plt.title('Point Cloud VAE train loss')
        plt.savefig(
            RESULT_PATH + model_name + "/" + 'loss_batchsize{}_epoch{}.png'.format(BATCH_SIZE, epoch)
        )

print("STARTING TRAINING")
print("DATASET PREP")

dataset = ShapeNet(root="./data/ShapeNet", categories=["Airplane"])
# snf.histogram(dataset, "airplanes", 20)
filtered = snf.filter_data(dataset, NB_VERTICES)
resampled = snf.simple_resample_to_minimum(filtered)
resampled = resampled[:100]
train, val = snf.random_split_ratio(resampled, 0.8)
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val, batch_size=BATCH_SIZE)

print("len resampled dataset: {}".format(len(resampled)))
print("size of data in resampled dataset: {}".format(resampled[5].pos.size()))
print("length train set: {}".format(len(train)))
print("length val set: {}".format(len(val)))

print("cuda available: {}".format(torch.cuda.is_available()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = MSELoss(reduction='mean')

# print("")
# print("TRAIN RANDOM POOL NETWORK")
#
# name = "pointcloudae_random"
# net = PointCoudAERandom()
#
# optimizer = Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# train_network(net, name, optimizer, loss_fn)
#
#
# print("")
# print("TRAIN FPS POOL NETWORK")
#
# name = "pointcloudeae_fps"
# net = PointCloudAE()
#
# optimizer = Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
# train_network(net, name, optimizer, loss_fn)

print("")
print("TRAIN RANDOM POOL NETWORK")

name = "simpelAE"
net = SimplePointCloudAE()

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of parameters: {}".format(params))

optimizer = Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
train_network(net, name, optimizer, loss_fn)

test = resampled[75]
out = net(test.pos)
loss = loss_fn(out, test.pos)

loss.backward()
optimizer.step()

print("")
print(out)
print(loss)

for param in net.parameters():
    print("")
    print(param.grad)
