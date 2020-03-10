import torch
import torch_geometric.nn as gnn
import meshplot as mp

import numpy as np

from dataset.primitives import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer


NB_EPOCHS = 100
NB_TESTS = 10
RADIUS = 0.22
RESULT_PATH = "D:/Documenten/Results/"
NAME = "SimpleRelativeLayerRadius/"

print("STARTING EVALUATING")
print("DATASET PREP")

dataset = PrimitiveShapes.generate_dataset(10, 2000)
data = dataset[0]
pos = data.pos
np_pos = pos.numpy()

sample_inds = gnn.fps(pos, ratio=0.05)
samples = pos[sample_inds]

# knn_cluster, knn_inds = gnn.knn(pos, samples, k=20)
# knn_points = pos[knn_inds]
# midpoints = samples[knn_cluster]
# relatives = knn_points - midpoints

rad_cluster, rad_inds = gnn.radius(pos, samples, r=RADIUS)
rad_points = pos[rad_inds]
rad_midpoints = samples[rad_cluster]
relatives = (rad_points - rad_midpoints) / RADIUS

"""
    VISUALISE LEARNING PROCESS
"""

# neighbourhood = knn_points[knn_cluster == 0]
# neighbourhood_np = neighbourhood.numpy()
# neighbourhood_rel = relatives[knn_cluster == 0]
# neigh_rel_np = neighbourhood_rel.numpy()
# cluster = knn_cluster[knn_cluster == 0]

neighbourhood = rad_points[rad_cluster == 0]
neighbourhood_np = neighbourhood.numpy()
neighbourhood_rel = relatives[rad_cluster == 0]
neigh_rel_np = neighbourhood_rel.numpy()
cluster = rad_cluster[rad_cluster == 0]

net = SimpleRelativeLayer(20, 20, 10, 5)

nb_evals = int(NB_EPOCHS / 5)

mp.offline()
plot = mp.subplot(
    np_pos, c=np_pos[:, 0],
    s=[nb_evals + 3, 1, 0], shading={"point_size": 0.3}
)
mp.subplot(
    neighbourhood_np, s=[nb_evals + 3, 1, 1],
    data=plot, shading={"point_size": 0.3}
)
mp.subplot(
    np_pos, c=np_pos[:, 0], s=[nb_evals + 3, 1, 2],
    data=plot, shading={"point_size": 0.3}
)
plot.rows[2][0].add_points(
    neighbourhood_np, c=np.tile([1.0, 0.5, 0.5], (neighbourhood_np.shape[0], 1)),
    shading={"point_size": 0.4}
)

for i in range(nb_evals):
    epoch = i * 5
    print("Epoch: {}".format(epoch))

    net.load_state_dict(
        torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(epoch))
    )
    net.eval()
    ae = net.ae

    relative_out = ae(neighbourhood_rel, cluster)
    output_np = relative_out.detach().numpy()

    points = np.concatenate((neigh_rel_np, output_np))
    colors = np.concatenate(
        (np.tile([1.0, 0, 0], (neigh_rel_np.shape[0], 1)),
         np.tile([0, 0, 1.0], (output_np.shape[0], 1)))
    )

    mp.subplot(
        points, c=colors, s=[nb_evals + 3, 1, i + 3],
        data=plot, shading={"point_size": 0.3}
    )

plot.save(RESULT_PATH + NAME + "visualisation")


"""
    VISUALISE FINAL MODEL ON DIFFERENT NEIGHBOURHOODS
"""

plot = mp.subplot(
    np_pos, c=np_pos[:, 0],
    s=[NB_TESTS * 2 + 1, 1, 0], shading={"point_size": 0.3}
)
net = SimpleRelativeLayer(20, 20, 10, 5)
net.load_state_dict(
    torch.load(RESULT_PATH + NAME + "model_epoch95.pt")
)
auto_encoder = net.ae

for i in range(NB_TESTS):
    print(i)
    # neigh = knn_points[knn_cluster == i]
    neigh = rad_points[rad_cluster == i]
    neigh_np = neigh.numpy()

    # neigh_rel = relatives[knn_cluster == i]
    neigh_rel = relatives[rad_cluster == i]
    neigh_rel_np = neigh_rel.numpy()

    # cluster = knn_cluster[knn_cluster == i]
    cluster = rad_cluster[rad_cluster == i]
    output = auto_encoder(neigh_rel, cluster)
    output_np = output.detach().numpy()

    mp.subplot(
        np_pos, c=np_pos[:, 0], s=[NB_TESTS * 2 + 1, 1, 2 * i + 1],
        data=plot, shading={"point_size": 0.3}
    )
    plot.rows[2 * i + 1][0].add_points(
        neigh_np, shading={"point_size": 0.4}
    )

    points = np.concatenate((neigh_rel_np, output_np))
    colors = np.concatenate(
        (np.tile([1.0, 0, 0], (neigh_rel_np.shape[0], 1)),
         np.tile([0, 0, 1.0], (output_np.shape[0], 1)))
    )
    mp.subplot(
        points, c=colors, s=[NB_TESTS * 2 + 1, 1, 2 * i + 2],
        data=plot, shading={"point_size": 0.3}
    )

plot.save(RESULT_PATH + NAME + "visualisation2")