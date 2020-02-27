import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.encoder import SimpleRelativeEncoder
from relative_layer.decoder import SimpleRelativeDecoder


class SimpleRelativeLayer(nn.Module):
    def __init__(self, nb_neighbours, feats1, feats2, feats3):
        super(SimpleRelativeLayer, self).__init__()
        self.enc = SimpleRelativeEncoder(feats1, feats2, feats3)
        self.dec = SimpleRelativeDecoder(feats3, feats2, feats1, nb_neighbours)

        self.nb_neighbours = nb_neighbours
        self.ratio = 1 / nb_neighbours

    def forward(self, points):

        # points = nb_points x 3

        sample_inds = gnn.fps(points, ratio=self.ratio)
        samples = points[sample_inds]
        # samples = (ratio * nb_points) x 3

        knn_cluster, knn_inds = gnn.knn(points, samples, k=self.nb_neighbours)
        knn_points = points[knn_inds]
        # knn_cluster = nb_point x 1
        # knn_points = nb_points x 3

        midpoints = samples[knn_cluster]
        relative = knn_points - midpoints
        # midpoints = nb_points x 3
        # relative = nb_points x 3

        encoded = self.enc(relative, knn_cluster)
        # encoded = (ratio * nb_points) x 5

        decoded = self.dec(encoded)
        # decoded = nb_points x 3

        dec_abs = decoded + midpoints
        # dec_abs = nb_points x 3

        resized_orig = knn_points.view(-1, self.nb_neighbours, 3)
        # resized_orig = (ratio * nb_points) x nb_neighbours x 3

        resized_deco = dec_abs.view(-1, self.nb_neighbours, 3)
        # resized_deco = (ratio * nb_points) x nb_neighbours x 3

        return resized_orig, resized_deco







