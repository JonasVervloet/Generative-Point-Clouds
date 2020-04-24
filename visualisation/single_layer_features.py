import torch
import torch_geometric.nn as gnn

import numpy as np
import meshplot as mp

from dataset.primitive_shapes import PrimitiveShapes
from relative_layer.simple_layer import SimpleRelativeLayer


RESULT_PATH = "D:/Documenten/Results/"
NAME = "LearningRate2/MoreExpressive/"
NB_EPOCHS = 90
NB_TESTS = 20

mp.offline()

print("DATASET LOADING")
dataset = PrimitiveShapes.generate_dataset(10, 2000)
net = SimpleRelativeLayer(20, 80, 40, 20, radius=0.22)
net.load_state_dict(
    torch.load(RESULT_PATH + NAME + "model_epoch{}.pt".format(NB_EPOCHS))
)
encoder = net.ae.enc
decoder = net.ae.dec

features = []
for data in dataset:
    pos = data.pos

    fps_inds = gnn.fps(pos, ratio=1/20)
    fps_points = pos[fps_inds]

    rad_cluster, rad_inds = gnn.radius(pos, fps_points, r=0.22)
    rad_points = pos[rad_inds]
    rad_midpoints = fps_points[rad_cluster]
    relatives = (rad_points - rad_midpoints) / 0.22

    encoded = encoder(relatives, rad_cluster)
    features.append(encoded)

feats_torch = torch.cat(features, dim=0)
print(feats_torch.size())

feats_min = torch.argmin(feats_torch, dim=0)
print(feats_min.size())

feats_max = torch.argmax(feats_torch, dim=0)
print(feats_max.size())

feats_mean = torch.mean(feats_torch, dim=0)
print(feats_mean.size())

for index in range(feats_mean.size(0)):
    maximum = feats_max[index]
    print(maximum)
    minimum = feats_min[index]
    print(minimum)
    if maximum == minimum:
        print("continue")
        continue
    interval = torch.arange(
        minimum,
        maximum,
        (maximum-minimum)/NB_TESTS
    )

    plot_created = False
    i = 0
    for feat_val in interval:
        feature = feats_mean.clone()
        feature[index] = feat_val

        decoded = decoder(feature)
        dec_np = decoded.detach().numpy()

        if not plot_created:
            plot = mp.subplot(
                dec_np, c=dec_np[:, 0],
                s=[NB_TESTS + 1, 1, i], shading={"point_size": 0.3}
            )
            plot_created = True
        else:
            mp.subplot(
                dec_np, c=dec_np[:, 0], data=plot,
                s=[NB_TESTS + 1, 1, i], shading={"point_size": 0.3}
            )

        i += 1

    plot.save(RESULT_PATH + NAME + "Feature visualization/" + "feature_visualization{}".format(index))


