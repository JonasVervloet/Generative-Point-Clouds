import torch.nn as nn
import torch_geometric.nn as gnn

from composed_layer.composed_encoder import ComposedEncoder
from composed_layer.composed_decoder import ComposedDecoder


class ComposedAutoEncoder(nn.Module):
    def __init__(self):
        super(ComposedAutoEncoder, self).__init__()

        self.enc = ComposedEncoder()
        self.dec = ComposedDecoder()

    def forward(self, points):
        print("AutoEncoder!")

        # points = nb_points x 3
        print(points.size())

        sample_inds = gnn.fps(points, ratio=0.05)
        samples = points[sample_inds]
        # RATIO_1 # => nb_samples = RATIO_1 * nb_points
        # RATIO_1 # => nb_neighs = 1/RATIO_1
        # sample_inds = nb_samples x 1
        # samples = nb_samples x 3
        print(sample_inds.size())
        print(samples.size())

        rad_cluster, rad_inds = gnn.radius(points, samples, r=0.3)
        rad_points = points[rad_inds]
        # RADIUS_1 #
        # rad_cluster = (nb_samples * nb_points1) x 1
        # rad_inds = (nb_samples * nb_points1) x 1
        # rad_points = (nb_samples * nb_points1) x 3
        print(rad_cluster.size())
        print(rad_inds.size())
        print(rad_points.size())

        midpoints = samples[rad_cluster]
        relative = (rad_points - midpoints) / 0.3
        # RADIUS_1 #
        # midpoints = (nb_samples * nb_points1) x 3
        # relative = (nb_samples * nb_points1) x 3
        print(midpoints.size())
        print(relative.size())

        samples2_inds = gnn.fps(samples, ratio=0.05)
        samples2 = samples[samples2_inds]
        # RATIO_2 # => nb_samples2 = RATIO_2 * nb_samples
        # RATIO_2 # => nb_neighs2 = 1/RATIO_2
        # samples2_inds = nb_samples2 x 1
        # samples2 = nb_samples2 x 3
        print(samples2_inds.size())
        print(samples2.size())

        rad2_cluster, rad2_inds = gnn.radius(samples, samples2, r=1.0)
        rad2_points = samples[rad2_inds]
        # RADIUS_2 #
        # rad2_cluster = (nb_samples2 * nb_points2) x 1
        # rad2_inds = (nb_samples2 * nb_points2) x 1
        # rad2_points = (nb_samples2 * nb_points2) x 3
        print(rad2_cluster.size())
        print(rad2_inds.size())
        print(rad2_points.size())

        midpoints2 = samples2[rad2_cluster]
        relative2 = (rad2_points - midpoints2) / 1.0
        # RADIUS_2 #
        # midpoints2 = (nb_samples2 * nb_points2) x 3
        # relative2 = (nb_samples2 * nb_points2) x 3
        print(midpoints2.size())
        print(relative2.size())

        print()
        encoding, feats = self.enc(relative, rad_cluster, relative2, rad2_inds, rad2_cluster)
        # encoding = nb_samples2 x 3
        # feats = nb_samples2 x 5
        print()
        print(encoding.size())
        print(feats.size())

        print()
        decoded, decoded2 = self.dec(encoding, feats)
        # decoded = (nb_samples2 * neighs2) x 3 = nb_samples x 3
        # decoded2 = (nb_samples * neighs) x 3 = points x 3
        print()
        print(decoded.size())
        print(decoded2.size())
        print()

        midpoints_out = (samples2.repeat_interleave(20, dim=0) + decoded) * 1.0
        # RATIO_2  # => nb_neighs2
        # RADIUS_2 #
        # midpoints_out = nb_samples x 3
        print(midpoints_out.size())

        points_out = (midpoints_out.repeat_interleave(20, dim=0) + decoded2) * 0.3
        # RATIO_1 # => nb_neighs
        # RADIUS_1 #
        # points_out = nb_points x 3
        print(points_out.size())

        return points_out
