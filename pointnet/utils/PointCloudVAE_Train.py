import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

from dataset.primitives import PrimitiveShapes
from PointNetAE import PointNetVAE
from LossFunctions import ChamferVAELoss

import matplotlib.pyplot as plt


NB_EPOCHS = 50
BATCH_SIZE = 5
RESULT_PATH = "D:/Documenten/Results/"
NAME = "VAEPointCloud/"

print("STARTING TRAINTNG")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(10, 2000)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

net = PointNetVAE()
optimizer = Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = ChamferVAELoss()

net.train()
losses = []

for epoch in range(NB_EPOCHS):
    print("epoch: {}".format(epoch))

    temp_loss = []
    i = 0
    for batch in loader:
        i += 1
        print("batch {}".format(i))

        optimizer.zero_grad()
        inp = batch.pos
        outp, z_mu, z_var = net(inp, batch.batch)

        loss = loss_fn(inp, outp, z_mu, z_var)
        loss.backward()
        optimizer.step()
        temp_loss.append(loss)

    losses.append(
        sum(temp_loss)/ (len(loader) * 20)
    )

    if epoch % 5 == 0:
        torch.save(
            net.state_dict(),
            RESULT_PATH + NAME + "model_epoch{}.pt".format(epoch)
        )
        plt.clf()
        x = range(len(losses))
        plt.plot(x, losses)
        plt.legend(['train loss'])
        plt.title('Simple Relative Layer train loss')
        plt.yscale('log')
        plt.savefig(
            RESULT_PATH + NAME + "loss_epoch{}.png".format(epoch)
        )
