import torch.nn as nn
from relative_layer.decoder import SimpleRelativeDecoder
from composed_layer.decoder import MiddleLayerDecoder


class ComposedDecoder(nn.Module):
    def __init__(self):
        super(ComposedDecoder, self).__init__()

        self.dec1 = MiddleLayerDecoder()
        self.dec2 = SimpleRelativeDecoder(5, 10, 20, 20)

    def forward(self, encoded, feats):

        # encoded = nb_cluster2 x 5
        # feats = nb_cluster2 x 25

        decoded, feats_dec = self.dec1(encoded, feats)
        # decoded = (nb_cluster2 * 20) x 3
        # feats_dec = (nb_cluster2 * 20) x 5

        decoded2 = self.dec2(feats_dec)
        # decoded2 = (nb_cluster * 20) x 3 = (nb_cluster2 * 400) x 3

        return decoded, decoded2