import torch_geometric.nn as gnn
import numpy as np
import meshplot as mp
from dataset.primitives import PrimitiveShapes as ps
from relative_layer.encoder import SimpleRelativeEncoder as sre
from relative_layer.decoder import SimpleRelativeDecoder as srd
from relative_layer.simple_layer import SimpleRelativeLayer

RESULT_PATH = "D:/Documenten/Results/Visualisations/"
mp.offline()

dataset = ps.generate_dataset(5, 3600, [True, False, False, False, False])
data = dataset[0]
print(data)
print(data.pos.size())

pos = data.pos
np_pos = pos.numpy()

samples_inds = gnn.fps(pos, ratio=1/25)
samples = pos[samples_inds]
np_samples = samples.numpy()

samples_inds2 = gnn.fps(samples, ratio=1/16)
samples2 = samples[samples_inds2]
np_samples2 = samples2.numpy()


fps_plot = mp.subplot(
    np_pos, c=np_pos[:, 0], s=[2, 3, 0],
    shading={"point_size": 0.4}
)
mp.subplot(
    np_samples, s=[2, 3, 1],
    data=fps_plot, shading={"point_size": 0.4}
)
mp.subplot(
    np_pos, c=np_pos[:, 0], s=[2, 3, 2],
    data=fps_plot, shading={"point_size": 0.4}
)
fps_plot.rows[0][2].add_points(
    np_samples, shading={"point_size": 0.5}
)

mp.subplot(
    np_samples, c=np_samples[:, 0], s=[2, 3, 3],
    data=fps_plot, shading={"point_size": 0.4}
)
mp.subplot(
    np_samples2, s=[2, 3, 4],
    data=fps_plot, shading={"point_size": 0.4}
)
mp.subplot(
    np_samples, c=np_samples[:, 0], s=[2, 3, 5],
    data=fps_plot, shading={"point_size": 0.4}
)
fps_plot.rows[1][2].add_points(
    np_samples2, shading={"point_size": 0.5}
)

fps_plot.save(RESULT_PATH + "fps_visualisation")


print(samples.size())
nb_samples = 10
knn_plot = mp.subplot(
    np_samples, c=np_samples[:, 0], s=[nb_samples, 3, 0],
    shading={"point_size": 0.4}
)

knn_cluster, knn_inds = gnn.knn(pos, samples, k=20)
for i in range(nb_samples):
    print(i)
    inds = knn_inds[knn_cluster == i]
    np_knns = pos[inds].numpy()
    print(np_knns.shape)
    print(np_knns)
    print()

    mp.subplot(
        np_pos, c=np_pos[:, 0], s=[nb_samples, 3, i * 3],
        data=knn_plot, shading={"point_size": 0.4}
    )
    mp.subplot(
        np_knns, s=[nb_samples, 3, i * 3 + 1],
        data=knn_plot, shading={"point_size": 0.4}
    )
    mp.subplot(
        np_pos, c=np_pos[:, 0], s=[nb_samples, 3, i * 3 + 2],
        data=knn_plot, shading={"point_size": 0.4}
    )
    knn_plot.rows[i][2].add_points(
        np_knns,
        shading={"point_size": 0.4}
    )
    knn_plot.rows[i][2].add_points(
        np.array([np_samples[i]], dtype=float),
        shading={"point_size": 0.5}
    )

knn_plot.save(RESULT_PATH + "knn_visualisation")

# encoder = sre(20, 10, 5)
# latent = encoder(data.pos[knn_inds], knn_cluster)
#
# decoder = srd(5, 10, 20, 20)
# out = decoder(latent)
# slayer = SimpleRelativeLayer(20, 20, 10, 5)
# slayer(data.pos)









