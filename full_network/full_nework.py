import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from full_network.full_encoder import FullEncoder
from full_network.full_decoder import FullDecoder


class FullNetwork(nn.Module):
    def __init__(self, ratio1=0.05, ratio2=0.05, nb_neighs3=5,
                 radius1=0.3, radius2=1.0, radius3=2.0,
                 feats1=5, feats2=25, feats3=45):
        super(FullNetwork, self).__init__()

        self.ratio1 = ratio1
        self.nb_neighs1 = int(1 / ratio1)
        self.ratio2 = ratio2
        self.nb_neighs2 = int(1 / ratio2)
        self.nb_neighs3 = nb_neighs3

        self.radius1 = radius1
        self.radius2 = radius2
        self.radius3 = radius3

        self.enc = FullEncoder(
            nb_feats1=feats1,
            nb_feats2=feats2,
            nb_feats3=feats3
        )
        self.dec = FullDecoder(
            nb_feats1=feats1,
            nb_feats2=feats2,
            nb_feats3=feats3,
            nb_neighs1=self.nb_neighs3,
            nb_neighs2=self.nb_neighs2,
            nb_neighs3=self.nb_neighs1
        )

    def forward(self, points, batch):
        print("FULL NETWORK")

        # points = nb_points x 3
        print(points.size())

        sample_inds = gnn.fps(points, ratio=self.ratio1, batch=batch)
        samples = points[sample_inds]
        batch_samples = batch[sample_inds]
        # nb_samples = ratio1 * nb_points
        # sample_inds = nb_samples x 1
        # samples = nb_samples x 3
        # batch_samples = nb_samples x 1
        print(sample_inds.size())
        print(samples.size())
        print(batch_samples.size())

        rad_cluster, rad_inds = gnn.radius(
            points, samples, r=self.radius1,
            batch_x=batch, batch_y=batch_samples
        )
        rad_points = points[rad_inds]
        # nb_points: nb of points within radius1 of the points in samples
        # rad_cluster = (nb_samples * nb_points1) x 1
        # rad_inds = (nb_samples * nb_points1) x 1
        # rad_points = (nb_samples * nb_points1) x 3
        print(rad_cluster.size())
        print(rad_inds.size())
        print(rad_points.size())

        midpoints = samples[rad_cluster]
        relative = (rad_points - midpoints) / self.radius1
        # midpoints = (nb_samples * nb_points1) x 3
        # relative = (nb_samples * nb_points1) x 3
        print(midpoints.size())
        print(relative.size())

        samples2_inds = gnn.fps(samples, ratio=self.ratio2, batch=batch_samples)
        samples2 = samples[samples2_inds]
        batch_samples2 = batch_samples[samples2_inds]
        # nb_samples2 = ratio2 * nb_samples
        # samples2_inds = nb_samples2 x 1
        # samples2 = nb_samples2 x 3
        print(samples2_inds.size())
        print(samples2.size())

        rad2_cluster, rad2_inds = gnn.radius(
            samples, samples2, r=self.radius2,
            batch_x=batch_samples, batch_y=batch_samples2
        )
        rad2_points = samples[rad2_inds]
        # nb_points2: nb of points within radius2 of the points in samples2
        # rad2_cluster = (nb_samples2 * nb_points2) x 1
        # rad2_inds = (nb_samples2 * nb_points2) x 1
        # rad2_points = (nb_samples2 * nb_points2) x 3
        print(rad2_cluster.size())
        print(rad2_inds.size())
        print(rad2_points.size())

        midpoints2 = samples2[rad2_cluster]
        relative2 = (rad2_points - midpoints2) / self.radius2
        # midpoints2 = (nb_samples2 * nb_points2) x 3
        # relative2 = (nb_samples2 * nb_points2) x 3
        print(midpoints2.size())
        print(relative2.size())

        relative3 = samples2 / self.radius3
        # relative3 = nb_samples2 x 3
        print(relative3.size())

        print()
        latent = self.enc(relative, rad_cluster, relative2, rad2_inds, rad2_cluster, relative3, batch_samples2)
        # latent = 1 x (nb_feats1 + nb_feats3)
        print()
        print(latent.size())

        print()
        decoded, decoded2, decoded3 = self.dec(latent)
        # decoded = nb_samples2 x 3
        # decoded2 = nb_samples x 3
        # decoded3 = nb_points x 3
        print()
        print(decoded.size())
        print(decoded2.size())
        print(decoded3.size())

        samples2_out = decoded * self.radius3
        # samples2_out = nb_samples2 x 3
        print(samples2_out.size())

        repeat2 = samples2_out.repeat_interleave(self.nb_neighs2, dim=0)
        # repeat2 = nb_samples x 3
        print(repeat2.size())

        samples_out = (repeat2 + decoded2) * self.radius2
        # samples_out = nb_samples x 3
        print(samples_out.size())

        repeat = samples_out.repeat_interleave(self.nb_neighs1, dim=0)
        # repeat = nb_points x 3
        print(repeat.size())

        points_out = (repeat + decoded3) * self.radius1
        # points_out = nb_points x 3
        print(points_out.size())

        return points_out
