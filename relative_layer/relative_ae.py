import torch
import torch.nn as nn
import torch_geometric

from relative_layer.encoder import SimpleRelativeEncoder
from relative_layer.decoder import SimpleRelativeDecoder


class RelativeAutoEncoder(nn.Module):
    def __init__(self, feats1, feats2, feats3, nb_neighbours, mean=False):
        super(RelativeAutoEncoder, self).__init__()
        self.enc = SimpleRelativeEncoder(feats1, feats2, feats3, mean)
        self.dec = SimpleRelativeDecoder(feats3, feats2, feats1, nb_neighbours)

    def forward(self, relative_points, cluster):

        # relative_points = nb_points x 3
        # cluster = nb_points x 1

        encoded = self.enc(relative_points, cluster)
        # encoded = (nb_neighbours^(-1) * nb_points) x 5

        decoded = self.dec(encoded)
        # decoded = nb_points x 3

        return decoded

    def set_encoder(self, encoder):
        self.enc = encoder

    def set_decoder(self, decoder):
        self.dec = decoder
