import torch
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes
from PointNetAE import PointNetVAE

NB_EPOCHS = 50
NB_INTERPOLS = 10
RESULT_PATH = "D:/Documenten/Results/"
NAME = "VAEPointCloud/"


print("STARTING EVALUATING")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(10, 2000)
data1 = dataset[0]
data2 = dataset[1]

net = PointNetVAE()

length = data1.pos.size(0)
batch = torch.tensor([0]).repeat(length)
output, z_mu, z_var = net(data1.pos, batch)
sample = torch.randn_like(z_mu)

nb_evals = int(len(dataset) / 5)

mp.offline()
np_pos = data1.pos.numpy()
plot = mp.subplot(
    np_pos, c=np_pos[:, 0], s=[nb_evals + 1, 1, 0], shading={"point_size": 0.5}
)

for i in range(nb_evals):
    epoch = i * 5
    print("Epoch: {}".format(epoch))

    net.load_state_dict(
        torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(epoch))
    )
    net.eval()

    output = net.decode(sample)
    print(output.size())

    out_np = output.detach().numpy()

    mp.subplot(
        out_np, c=out_np[:, 0], s=[nb_evals + 1, 1, i + 1],
        data=plot, shading={"point_size": 0.5}
    )

plot.save(
    RESULT_PATH + NAME + "eval"
)

net.load_state_dict(
    torch.load(RESULT_PATH + NAME + "model_epoch45.pt")
)

latent1 = net.evaluate(data1.pos, batch)
latent2 = net.evaluate(data2.pos, batch)
diff = latent2 - latent1
step = diff / NB_INTERPOLS

data1_np = data1.pos.numpy()
data2_np = data2.pos.numpy()

plot = mp.subplot(
    data1_np, c=data1_np[:, 0], s=[NB_INTERPOLS + 2, 1, 0], shading={"point_size": 0.5}
)

for i in range(NB_INTERPOLS):
    print("Interpol: {}".format(i))
    latent = latent1 + i * step
    outp = net.decode(latent)
    outp_np = outp.detach().numpy()

    mp.subplot(
        outp_np, c=outp_np[:, 0], s=[NB_INTERPOLS, 1, i + 1],
        data=plot, shading={"point_size": 0.5}
    )

mp.subplot(
    data2_np, c=data2_np[:, 0], s=[NB_INTERPOLS + 2, 1, NB_INTERPOLS + 1],
    data=plot, shading={"point_size": 0.5}
)

print(plot)
print(plot.rows)

plot.save(
    RESULT_PATH + NAME + "interpolation"
)

