import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from full_network.full_encoder import FullEncoder
from full_network.full_decoder import FullDecoder


class FullNetworkVAE(nn.Module):
    def __init__(self, ratio1=1/25, ratio2=1/16, nb_neighs3=9,
                 radius1=0.23, radius2=1.1, radius3=2.0,
                 feats1=20, feats2=30, feats3=80):
        super(FullNetworkVAE, self).__init__()

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

        self.fc_mu = nn.Linear(feats3 + feats1, feats3 + feats1)
        self.fc_sigma = nn.Linear(feats3 + feats1, feats3 + feats1)

        self.dec = FullDecoder(
            nb_feats1=feats1,
            nb_feats2=feats2,
            nb_feats3=feats3,
            nb_neighs1=self.nb_neighs3,
            nb_neighs2=self.nb_neighs2,
            nb_neighs3=self.nb_neighs1
        )

    def forward(self, points, batch):

        # points = nb_points x 3
        # batch = nb_points x 3
        # nb_batch: max nb in batch => nb of clouds in this batch

        sample_inds = gnn.fps(points, ratio=self.ratio1, batch=batch)
        samples = points[sample_inds]
        batch_samples = batch[sample_inds]
        # nb_samples = ratio1 * nb_points
        # sample_inds = nb_samples x 1
        # samples = nb_samples x 3
        # batch_samples = nb_samples x 1

        rad_cluster, rad_inds = gnn.radius(
            points, samples, r=self.radius1,
            batch_x=batch, batch_y=batch_samples
        )
        rad_points = points[rad_inds]
        # nb_points: nb of points within radius1 of the points in samples
        # rad_cluster = (nb_samples * nb_points1) x 1
        # rad_inds = (nb_samples * nb_points1) x 1
        # rad_points = (nb_samples * nb_points1) x 3

        midpoints = samples[rad_cluster]
        relative = (rad_points - midpoints) / self.radius1
        # midpoints = (nb_samples * nb_points1) x 3
        # relative = (nb_samples * nb_points1) x 3

        samples2_inds = gnn.fps(samples, ratio=self.ratio2, batch=batch_samples)
        samples2 = samples[samples2_inds]
        batch_samples2 = batch_samples[samples2_inds]
        # nb_samples2 = ratio2 * nb_samples
        # samples2_inds = nb_samples2 x 1
        # samples2 = nb_samples2 x 3

        rad2_cluster, rad2_inds = gnn.radius(
            samples, samples2, r=self.radius2,
            batch_x=batch_samples, batch_y=batch_samples2
        )
        rad2_points = samples[rad2_inds]
        # nb_points2: nb of points within radius2 of the points in samples2
        # rad2_cluster = (nb_samples2 * nb_points2) x 1
        # rad2_inds = (nb_samples2 * nb_points2) x 1
        # rad2_points = (nb_samples2 * nb_points2) x 3

        midpoints2 = samples2[rad2_cluster]
        relative2 = (rad2_points - midpoints2) / self.radius2
        # midpoints2 = (nb_samples2 * nb_points2) x 3
        # relative2 = (nb_samples2 * nb_points2) x 3

        relative3 = samples2 / self.radius3
        # relative3 = nb_samples2 x 3

        latent = self.enc(relative, rad_cluster, relative2, rad2_inds, rad2_cluster, relative3, batch_samples2)
        # latent = nb_batch x (nb_feats1 + nb_feats3)

        mean = self.fc_mu(latent)
        log_variance = self.fc_sigma(latent)
        std = torch.exp(log_variance*0.5)
        eps = torch.randn_like(std)

        x_sample = eps.mul(std).add(mean)

        decoded, decoded2, decoded3 = self.dec(x_sample)
        # decoded = nb_samples2 x 3
        # decoded2 = nb_samples x 3
        # decoded3 = nb_points x 3

        samples2_out = decoded * self.radius3
        # samples2_out = nb_samples2 x 3

        repeat2 = samples2_out.repeat_interleave(self.nb_neighs2, dim=0)
        # repeat2 = nb_samples x 3

        samples_out = repeat2 + (decoded2 * self.radius2)
        # samples_out = nb_samples x 3

        repeat = samples_out.repeat_interleave(self.nb_neighs1, dim=0)
        # repeat = nb_points x 3

        points_out = repeat + (decoded3 * self.radius1)
        # points_out = nb_points x 3

        batch_range = torch.arange(torch.max(batch) + 1)
        batch_out = batch_range.repeat_interleave(
            self.nb_neighs1 * self.nb_neighs2 * self.nb_neighs3
        )
        # batch_out = (nb_batch * nb_neighs1 * nb_neighs2 * nb_neighs3) x 1

        return points_out, batch_out, mean, log_variance

    def evaluate(self, points, batch):
        # points = nb_points x 3
        # batch = nb_points x 3
        # nb_batch: max nb in batch => nb of clouds in this batch

        sample_inds = gnn.fps(points, ratio=self.ratio1, batch=batch)
        samples = points[sample_inds]
        batch_samples = batch[sample_inds]
        # nb_samples = ratio1 * nb_points
        # sample_inds = nb_samples x 1
        # samples = nb_samples x 3
        # batch_samples = nb_samples x 1

        rad_cluster, rad_inds = gnn.radius(
            points, samples, r=self.radius1,
            batch_x=batch, batch_y=batch_samples
        )
        rad_points = points[rad_inds]
        # nb_points: nb of points within radius1 of the points in samples
        # rad_cluster = (nb_samples * nb_points1) x 1
        # rad_inds = (nb_samples * nb_points1) x 1
        # rad_points = (nb_samples * nb_points1) x 3

        midpoints = samples[rad_cluster]
        relative = (rad_points - midpoints) / self.radius1
        # midpoints = (nb_samples * nb_points1) x 3
        # relative = (nb_samples * nb_points1) x 3

        samples2_inds = gnn.fps(samples, ratio=self.ratio2, batch=batch_samples)
        samples2 = samples[samples2_inds]
        batch_samples2 = batch_samples[samples2_inds]
        # nb_samples2 = ratio2 * nb_samples
        # samples2_inds = nb_samples2 x 1
        # samples2 = nb_samples2 x 3

        rad2_cluster, rad2_inds = gnn.radius(
            samples, samples2, r=self.radius2,
            batch_x=batch_samples, batch_y=batch_samples2
        )
        rad2_points = samples[rad2_inds]
        # nb_points2: nb of points within radius2 of the points in samples2
        # rad2_cluster = (nb_samples2 * nb_points2) x 1
        # rad2_inds = (nb_samples2 * nb_points2) x 1
        # rad2_points = (nb_samples2 * nb_points2) x 3

        midpoints2 = samples2[rad2_cluster]
        relative2 = (rad2_points - midpoints2) / self.radius2
        # midpoints2 = (nb_samples2 * nb_points2) x 3
        # relative2 = (nb_samples2 * nb_points2) x 3

        relative3 = samples2 / self.radius3
        # relative3 = nb_samples2 x 3

        latent = self.enc(relative, rad_cluster, relative2, rad2_inds, rad2_cluster, relative3, batch_samples2)
        # latent = nb_batch x (nb_feats1 + nb_feats3)

        mean = self.fc_mu(latent)

        decoded, decoded2, decoded3 = self.dec(mean)
        # decoded = nb_samples2 x 3
        # decoded2 = nb_samples x 3
        # decoded3 = nb_points x 3

        samples2_out = decoded * self.radius3
        # samples2_out = nb_samples2 x 3

        repeat2 = samples2_out.repeat_interleave(self.nb_neighs2, dim=0)
        # repeat2 = nb_samples x 3

        samples_out = repeat2 + (decoded2 * self.radius2)
        # samples_out = nb_samples x 3

        repeat = samples_out.repeat_interleave(self.nb_neighs1, dim=0)
        # repeat = nb_points x 3

        points_out = repeat + (decoded3 * self.radius1)
        # points_out = nb_points x 3

        batch_range = torch.arange(torch.max(batch) + 1)
        batch_out = batch_range.repeat_interleave(
            self.nb_neighs1 * self.nb_neighs2 * self.nb_neighs3
        )
        # batch_out = (nb_batch * nb_neighs1 * nb_neighs2 * nb_neighs3) x 1

        return samples2_out, samples_out, points_out, mean
