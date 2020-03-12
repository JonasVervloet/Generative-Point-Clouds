import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

from dataset.primitives import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer
from LossFunctions import ChamferDistLoss

import matplotlib.pyplot as plt

NB_EPOCHS = 200
RESULT_PATH = "D:/Documenten/Results/"
NAME = "LearningRate/"
START_LR = 0.001
LR_NB = 5


print("STARTING TRAINTNG")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(20, 2000)
loader = DataLoader(dataset=dataset, batch_size=1)
print(len(loader))

loss_fn = ChamferDistLoss()

for i in range(LR_NB):
    learning_rate = START_LR / max(1, (i * 10))
    path = RESULT_PATH + NAME + "LearningRate{}/".format(round(learning_rate * 100000))
    print(learning_rate)

    net = SimpleRelativeLayer(20, 20, 10, 5, mean=False)
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    net.train()
    losses = []

    for epoch in range(NB_EPOCHS):
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
            x = range(len(losses))
            plt.plot(x, losses)
            plt.legend(['train loss'])
            plt.title('Simple Relative Layer train loss')
            plt.yscale('log')
            plt.savefig(
                path + "loss_epoch{}.png".format(epoch)
            )









