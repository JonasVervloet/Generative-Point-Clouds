import torch_geometric.nn as gnn

from composed_layer.encoder import ComposeLayerEncoder
from relative_layer.encoder import SimpleRelativeEncoder
from dataset.primitives import PrimitiveShapes


encoder = SimpleRelativeEncoder(20, 10, 5)
encoder2 = ComposeLayerEncoder(25)

dataset = PrimitiveShapes.generate_dataset(1, 2000)
points = dataset[0].pos
print(points.size())

sample_inds = gnn.fps(points, ratio=0.05)
samples = points[sample_inds]

rad_cluster, rad_inds = gnn.radius(points, samples, r=0.3)
rad_points = points[rad_inds]
print(rad_points.size())

midpoints = samples[rad_cluster]
relative = (rad_points - midpoints) / 0.3

feats_out = encoder(relative, rad_cluster)
print(feats_out.size())

samples2_inds = gnn.fps(samples, ratio=0.05)
samples2 = samples[samples2_inds]
print(samples2.size())

rad2_cluster, rad2_inds = gnn.radius(samples, samples2, r=1.0)
rad2_points = samples[rad2_inds]
rad2_feats = feats_out[rad2_inds]
print(rad2_points.size())
print(rad2_feats.size())

midpoints2 = samples2[rad2_cluster]
relative2 = (rad2_points - midpoints2) / 1.0
print(midpoints2.size())
print(relative2.size())

print()

encoder2(relative2, rad2_feats, rad2_cluster)











