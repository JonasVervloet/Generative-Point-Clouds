import torch
from torch_geometric.data import DataLoader

import meshplot as mp

from dataset.primitives import PrimitiveShapes
from full_network.full_nework import FullNetwork


RESULT_PATH = "D:/Documenten/Results/"
NAME = "FullNetwork/LearningRate100/"
BATCH_SIZE = 5
EPOCH = 100

dataset = PrimitiveShapes.generate_dataset(1, 3600)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

net = FullNetwork()
net.load_state_dict(
        torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(EPOCH))
    )
net.eval()

dummy = dataset[0].pos.numpy()

mp.offline()
plot = mp.subplot(
    dummy, c=dummy[:, 0],
    s=[BATCH_SIZE*2 + 1, 1, 0], shading={"point_size": 0.3}
)

for batch in loader:
    print("batch")

    points_in = batch.pos
    batch_in = batch.batch

    points_out, batch_out = net(points_in, batch_in)

    for i in range(BATCH_SIZE):
        points_in_np = points_in[batch_in == i].numpy()
        points_out_np = points_out[batch_out == i].detach().numpy()

        mp.subplot(
            points_in_np, c=points_in_np[:, 0], data=plot,
            s=[BATCH_SIZE*2 + 1, 1, i*2 + 1], shading={"point_size": 0.3}
        )
        mp.subplot(
            points_out_np, c=points_out_np[:, 0], data=plot,
            s=[BATCH_SIZE*2 + 1, 1, i*2 + 2], shading={"point_size": 0.3}
        )

plot.save(RESULT_PATH + NAME + "visualisation")

sphere = PrimitiveShapes.generate_dataset(1, 3600, [True, False, False, False, False])[0]
cube = PrimitiveShapes.generate_dataset(1, 3600, [False, True, False, False, False])[0]
cylinder = PrimitiveShapes.generate_dataset(1, 3600, [False, False, True, False, False])[0]
pyramid = PrimitiveShapes.generate_dataset(1, 3600, [False, False, False, True, False])[0]
torus = PrimitiveShapes.generate_dataset(1, 3600, [False, False, False, False, True])[0]

data_list = [[sphere, "sphere"], [cube, "cube"], [cylinder, "cylinder"], [pyramid, "pyramid"], [torus, "torus"]]

for data in data_list:
    shape = data[0]
    shape_name = data[1]

    pos = shape.pos
    pos_np = pos.numpy()

    batch = torch.zeros(pos.size(0), dtype=torch.int64)

    samples2, samples, points = net.evaluate(pos, batch)
    samples2_np = samples2.detach().numpy()
    samples_np = samples.detach().numpy()
    points_np = points.detach().numpy()

    plot = mp.subplot(
        pos_np, c=pos_np[:, 0],
        s=[4, 1, 0], shading={"point_size": 0.3}
    )

    mp.subplot(
        samples2_np, c=samples2_np[:, 0], data=plot,
        s=[4, 1, 1], shading={"point_size": 0.3}
    )
    mp.subplot(
        samples_np, c=samples_np[:, 0], data=plot,
        s=[4, 1, 2], shading={"point_size": 0.3}
    )
    mp.subplot(
        points_np, c=points_np[:, 0], data=plot,
        s=[4, 1, 3], shading={"point_size": 0.3}
    )

    plot.save(RESULT_PATH + NAME + shape_name)







