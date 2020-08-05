import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.relative_ae import RelativeAutoEncoder


class SimpleRelativeLayer2(nn.Module):
    def __init__(self, nb_neighbours, nb_neighbours2, feats1, feats2, feats3, radius=1.3, mean=False):
        super(SimpleRelativeLayer2, self).__init__()
        self.ae = RelativeAutoEncoder(feats1, feats2, feats3, nb_neighbours, mean)

        self.nb_neighbours = nb_neighbours
        self.ratio = 1 / nb_neighbours
        self.neighs2 = nb_neighbours2
        self.ratio2 = 1 / nb_neighbours2
        self.radius = radius

    def forward(self, points):

        # points = nb_points x 3

        sample_inds = gnn.fps(points, ratio=self.ratio)
        samples = points[sample_inds]
        # samples = (ratio * nb_points) x 3

        # samples_inds2 =
        # samples2 =
        samples_inds2 = gnn.fps(samples, ratio=self.ratio2)
        samples2 = samples[samples_inds2]

        rad_cluster, rad_inds = gnn.radius(samples, samples2, r=self.radius)
        rad_points = samples[rad_inds]

        # midpoints = samples[knn_cluster]
        midpoints = samples2[rad_cluster]
        # relative = knn_points - midpoints
        relative = (rad_points - midpoints) / self.radius
        # midpoints = nb_points x 3
        # relative = nb_points x 3

        relative_out = self.ae(relative, rad_cluster)

        # range = nb_cluster x 1
        # range_inds = (nb_cluster * nb_neighbours) x 1
        # midpoints_out = (nb_cluster * nb_neighbours) x 3
        range = torch.arange(0, samples2.size(0))
        range_inds = range.repeat_interleave(self.nb_neighbours)
        midpoints_out = samples2[range_inds]

        dec_abs = (relative_out * self.radius) + midpoints_out
        # dec_abs = nb_points x 3

        resized_deco = dec_abs.view(-1, self.nb_neighbours, 3)
        # resized_deco = (ratio * nb_points) x nb_neighbours x 3

        return rad_points, rad_cluster, resized_deco

    def set_encoder(self, encoder):
        self.ae.set_encoder(encoder)

    def set_decoder(self, decoder):
        self.ae.set_decoder(decoder)

