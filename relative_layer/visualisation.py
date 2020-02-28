import torch
import torch_geometric.nn as gnn
import meshplot as mp

from dataset.primitives import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer


NB_EPOCHS = 50
RESULT_PATH = "D:/Documenten/Results/"
NAME = "SimpleRelativeLayer/"

print("STARTING EVALUATING")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(10, 2000)
data = dataset[0]
pos = data.pos
np_pos = pos.numpy()

sample_inds = gnn.fps(pos, ratio=0.05)
samples = pos[sample_inds]
knn_cluster, knn_inds = gnn.knn(pos, samples, k=20)
knn_sample_inds = knn_inds[knn_cluster == 0]
knn_samples = pos[knn_sample_inds]
knn_samples_np = knn_samples.numpy()
print(knn_samples.size())

net = SimpleRelativeLayer(20, 20, 10, 5)

nb_evals = int(NB_EPOCHS / 5)

mp.offline()
plot = mp.subplot(
    np_pos, c=np_pos[:, 0],
    s=[nb_evals + 3, 1, 0], shading={"point_size": 0.3}
)
mp.subplot(
    knn_samples_np, s=[nb_evals + 3, 1, 1],
    data=plot, shading={"point_size": 0.3}
)
mp.subplot(
    np_pos, c=np_pos[:, 0], s=[nb_evals + 3, 1, 2],
    data=plot, shading={"point_size": 0.3}
)
plot.rows[2][0].add_points(
    knn_samples_np, shading={"point_size": 0.4}
)

for i in range(nb_evals):
    epoch = i * 5
    print("Epoch: {}".format(epoch))

    net.load_state_dict(
        torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(epoch))
    )
    net.eval()

    orig, outp = net(knn_samples)
    output_np = outp[0].detach().numpy()
    print(output_np.shape)

    mp.subplot(
        output_np, c=output_np[:, 0], s=[nb_evals + 3, 1, i + 3],
        data=plot, shading={"point_size": 0.5}
    )

plot.save(RESULT_PATH + NAME + "visualisation")

