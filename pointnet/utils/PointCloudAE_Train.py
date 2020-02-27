import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from dataset.functionality import ShapeNetFunctionality as snf
from dataset.primitive_shapes import PrimitiveShapes as ps
from PointNetAE import PointNetAE
from LossFunctions import ChamferDistLoss
from RandomPermutation import RandomPermutation
import meshplot as mp

import numpy as np
import matplotlib.pyplot as plt

NUMBER_EPOCHS = 80
BATCH_SIZE = 5
NB_VERTICES = 500
RESULT_PATH = "C:/Users/vervl/OneDrive/Documenten/GitHub/Generative-Mesh-Models/result/"


def batch_loss(input, output, loss_function):
    reshaped_input = snf.reshape_batch(input, NB_VERTICES)
    reshaped_output = snf.reshape_batch(output, NB_VERTICES)
    return loss_function(reshaped_output, reshaped_input)


def make_train_step(model, loss_function, opt):
    def train_step(x):
        model.train()
        opt.zero_grad()
        print(x.pos.size())
        print(x.batch.size())
        # out, z_mu, z_var = model(x.pos, x.batch)
        out = model(x.pos, x.batch)
        chamf_loss = batch_loss(x.pos, out, loss_function)
        # kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        # loss = chamf_loss + kl_loss
        loss = chamf_loss
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


def train_network(model, model_name, opt, loss_function, train_loader, val_loader):
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
            model.state_dict(), RESULT_PATH + model_name + "/" + "model_batchsize{}_epoch{}.pt".format(BATCH_SIZE, epoch)
        )
        plt.clf()
        x = range(len(losses))
        plt.plot(x, losses, x, val_losses)
        plt.legend(['train loss', 'val loss'])
        plt.title('Point Cloud VAE train loss')
        plt.savefig(
            RESULT_PATH + model_name + "/" + 'loss_batchsize{}_epoch{}.png'.format(BATCH_SIZE, epoch)
        )


def train_without_val(model, model_name, opt, loss_function, train_loader):
    train_step = make_train_step(model, loss_function, opt)
    losses = []

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
            sum(temp_loss) / 30
        )
        print("train loss: {}".format(losses[epoch]))
        print("")
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                RESULT_PATH + model_name + "/" + "model_batchsize{}_epoch{}.pt".format(BATCH_SIZE, epoch)
            )
            plt.clf()
            x = range(len(losses))
            plt.plot(x, losses)
            plt.legend(['train loss'])
            plt.title('Point Cloud VAE train loss')
            plt.yscale('log')
            plt.savefig(
                RESULT_PATH + model_name + "/" + 'loss_batchsize{}_epoch{}.png'.format(BATCH_SIZE, epoch)
            )


def train_one_sample(model, name, opt, loss_function, test):
    losses = []
    model.train()

    for epoch in range(NUMBER_EPOCHS):
        print("epoch: {}".format(epoch))
        print("TRAIN")
        opt.zero_grad()
        out = model(test.pos)
        loss = loss_function(test, out)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print("train loss: {}".format(losses[epoch]))
        print("")
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                RESULT_PATH + name + "/" + "model_batchsize{}_epoch{}.pt".format(BATCH_SIZE, epoch)
            )
            plt.clf()
            x = range(len(losses))
            plt.plot(x, losses)
            plt.legend(['train loss'])
            plt.title('Point Cloud VAE train loss')
            plt.yscale('log')
            plt.savefig(
                RESULT_PATH + name + "/" + 'loss_batchsize{}_epoch{}.png'.format(BATCH_SIZE, epoch)
            )


def eval_model(model, name, nb_epochs, test):
    mp.offline()
    pos = test.pos
    length = pos.size(0)
    print(length)
    batch = torch.tensor([0]).repeat(length)
    print(np.max(pos.detach().numpy(), axis=0))
    np_pos = pos.numpy()
    nb_images = (nb_epochs // 5) + 1
    plot = mp.subplot(
        np_pos, c=np_pos[:, 0], s=[nb_images, 1, 0], shading={"point_size": 0.5}
    )
    for epoch in range(nb_epochs):
        print(epoch)
        if epoch % 5 == 0:
            model.load_state_dict(
                torch.load(RESULT_PATH + name + "/" + "model_batchsize{}_epoch{}.pt".format(BATCH_SIZE, epoch))
            )
            model.eval()
            # result, mu, var = model(pos, batch)
            result = model(pos, batch)
            result = result.detach().numpy()
            print(np.max(result, axis=0))
            mp.subplot(
                result, c=result[:, 0], s=[nb_images, 1, (epoch // 5) + 1],
                data=plot, shading={"point_size": 0.5}
            )
    plot.save(RESULT_PATH + name + "/" + "eval")


print("STARTING TRAINING")
print("DATASET PREP")

print("cuda available: {}".format(torch.cuda.is_available()))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = ChamferDistLoss()

dataset = ShapeNet(root="C:/Users/vervl/Data/ShapeNet/utils",
                   train=True,
                   categories=["Airplane"],
                   transform=T.Compose([T.FixedPoints(2300), RandomPermutation()]))
print(dataset[0])

# snf.histogram(dataset, "airplanes", 30)
# filtered = snf.filter_data(dataset, NB_VERTICES)
# resampled = snf.simple_resample_to_minimum(filtered)
# resampled = resampled[:50]
# train = filtered[:30]
# train, val = snf.random_split_ratio(resampled, 0.8)
# train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE)
# val_loader = DataLoader(dataset=val, batch_size=BATCH_SIZE)

# train = spacu.generate_sphere_and_cubes_dataset(200)
# train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE)

train = ps.generate_dataset(50, 600)
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE)

print("len resampled dataset: {}".format(len(train)))
print("size of data in resampled dataset: {}".format(train[0].pos.size()))
print("length train set: {}".format(len(train)))
# print("length val set: {}".format(len(val)))

print("")
print("TRAIN PointNet AE NETWORK")

name = "PS_PointCloudAE"
net = PointNetAE()
# net.to(device)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("number of parameters: {}".format(params))

# optimizer = Adam(net.parameters(), lr=0.1, weight_decay=5e-4)
# train_without_val(net, name, optimizer, loss_fn, train_loader)

eval_model(PointNetAE(), name, NUMBER_EPOCHS, train[0])

# mp.offline()
# datalist = ps.generate_dataset(2, 500)
# nb_images = 10
# print(datalist[0])
# np_pos = datalist[0].pos.numpy()
# plot = mp.subplot(
#     np_pos, c=np_pos[:, 0], s=[nb_images, 1, 0], shading={"point_size": 0.5}
# )
# for i in range(1, nb_images):
#     np_pos = datalist[i].pos.numpy()
#     print(np.max(np_pos, axis=0))
#     mp.subplot(
#         np_pos, c=np_pos[:, 0], s=[nb_images, 1, i],
#         data=plot, shading={"point_size": 0.5}
#     )
# plot.save(RESULT_PATH + "test_dataset")

