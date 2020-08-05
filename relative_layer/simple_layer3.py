import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.relative_ae import RelativeAutoEncoder


class SimpleRelativeLayer3(nn.Module):
    def __init__(self, nb_neighbours, nb_neighbours2, nb_neighbours3, feats1, feats2, feats3, radius=2.0, mean=False):
        super(SimpleRelativeLayer3, self).__init__()

        self.ae = RelativeAutoEncoder(feats1, feats2, feats3, nb_neighbours3, mean)

        self.neighs = nb_neighbours
        self.ratio = 1 / nb_neighbours

        self.neighs2 = nb_neighbours2
        self.ratio2 = 1 / nb_neighbours2

        self.neighs3 = nb_neighbours3

        self.radius = radius

    def forward(self, points, batch):

        # points = nb_points x 3

        # samples = (ratio * nb_points) x 3
        # samples_batch =
        sample_inds = gnn.fps(points, batch=batch, ratio=self.ratio)
        samples = points[sample_inds]
        samples_batch = batch[sample_inds]

        # samples_inds2 =
        # samples2 =
        samples_inds2 = gnn.fps(samples, batch=samples_batch, ratio=self.ratio2)
        samples2 = samples[samples_inds2]
        samples2_batch = samples_batch[samples_inds2]

        # relative = nb_points x 3
        relative = samples2 / self.radius

        relative_out = self.ae(relative, samples2_batch)

        dec_abs = relative_out * self.radius
        # dec_abs = nb_points x 3

        resized_deco = dec_abs.view(-1, self.neighs3, 3)
        # resized_deco = (ratio * nb_points) x nb_neighbours x 3

        return samples2, samples2_batch, resized_deco

    def set_encoder(self, encoder):
        self.ae.set_encoder(encoder)

    def set_decoder(self, decoder):
        self.ae.set_decoder(decoder)

