import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.relative_ae import RelativeAutoEncoder


class SimpleRelativeLayer(nn.Module):
    def __init__(self, nb_neighbours, feats1, feats2, feats3, radius=0.22, mean=False):
        super(SimpleRelativeLayer, self).__init__()
        self.ae = RelativeAutoEncoder(feats1, feats2, feats3, nb_neighbours, mean)

        self.nb_neighbours = nb_neighbours
        self.ratio = 1 / nb_neighbours
        self.radius = radius

    def forward(self, points):

        # points = nb_points x 3

        sample_inds = gnn.fps(points, ratio=self.ratio)
        samples = points[sample_inds]
        # samples = (ratio * nb_points) x 3

        # knn_cluster, knn_inds = gnn.knn(points, samples, k=self.nb_neighbours)
        # knn_points = points[knn_inds]
        # knn_cluster = nb_point x 1
        # knn_points = nb_points x 3

        rad_cluster, rad_inds = gnn.radius(points, samples, r=self.radius)
        rad_points = points[rad_inds]

        # midpoints = samples[knn_cluster]
        midpoints = samples[rad_cluster]
        # relative = knn_points - midpoints
        relative = (rad_points - midpoints) / self.radius
        # midpoints = nb_points x 3
        # relative = nb_points x 3

        relative_out = self.ae(relative, rad_cluster)

        range = torch.arange(0, samples.size(0))
        range_inds = range.repeat_interleave(self.nb_neighbours)
        midpoints_out = samples[range_inds]

        dec_abs = (relative_out * self.radius) + midpoints_out
        # dec_abs = nb_points x 3

        resized_deco = dec_abs.view(-1, self.nb_neighbours, 3)
        # resized_deco = (ratio * nb_points) x nb_neighbours x 3

        return rad_points, rad_cluster, resized_deco







