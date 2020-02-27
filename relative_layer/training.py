import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

from dataset.primitive_shapes import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer
from LossFunctions import ChamferDistLoss

import matplotlib.pyplot as plt

NB_EPOCHS = 50
RESULT_PATH = "D:/Documenten/Results/"


print("STARTING TRAINTNG")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(10, 2000)
loader = DataLoader(dataset=dataset, batch_size=1)
print(len(loader))

net = SimpleRelativeLayer(20, 20, 10, 5)
optimizer = Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = ChamferDistLoss()

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

        origin, output = net(batch.pos)
        loss = loss_fn(origin, output)
        loss.backward()
        optimizer.step()
        temp_loss.append(loss)

    losses.append(
        sum(temp_loss) / (len(loader) * 20)
    )

    if epoch % 5 == 0:
        torch.save(
            net.state_dict(),
            RESULT_PATH + "SimpleRelativeLayer/" + "model_epoch{}.pt".format(epoch)
        )
        plt.clf()
        x = range(len(losses))
        plt.plot(x, losses)
        plt.legend(['train loss'])
        plt.title('Simple Relative Layer train loss')
        plt.yscale('log')
        plt.savefig(
            RESULT_PATH + "SimpleRelativeLayer/" + "loss_epoch{}.png".format(epoch)
        )









