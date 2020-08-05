import torch
import torch.nn as nn

from relative_layer.decoder import SimpleRelativeDecoder
from composed_layer.decoder import MiddleLayerDecoder


class FullDecoder(nn.Module):
    def __init__(self, nb_feats1=20, nb_feats2=30, nb_feats3=80,
                 nb_neighs1=9, nb_neighs2=16, nb_neighs3=25):
        super(FullDecoder, self).__init__()

        self.dec1 = MiddleLayerDecoder(
            nb_feats_in2=nb_feats3,
            nb_feats_out=nb_feats1+nb_feats2,
            nb_outputs=nb_neighs1
        )
        self.dec2 = MiddleLayerDecoder(
            nb_feats_in2=nb_feats2,
            nb_feats_out=nb_feats1,
            nb_outputs=nb_neighs2
        )
        self.dec3 = SimpleRelativeDecoder(
            nb_feats1,
            40,
            80,
            nb_neighs3
        )

        self.feats1 = nb_feats1

    def forward(self, latent):

        # latent = nb_batch x (nb_feats1 + nb_feats3)

        latent_neighb = latent[:, :self.feats1]
        latent_feats = latent[:, self.feats1:]
        # latent_neighb = nb_batch x nb_feats1
        # latent_feats = nb_batch x nb_feats3

        decoded, concat = self.dec1(latent_neighb, latent_feats)
        # nb_cluster = nb_neighs1 * nb_batch
        # decoded = nb_cluster x 3
        # concat = nb_cluster x (nb_feats1 + nb_feats2)

        neighb = concat[:, :self.feats1]
        feats = concat[:, self.feats1:]
        # neighb = nb_cluster x nb_feats1
        # feats = nb_cluster x nb_feats2

        decoded2, feats2 = self.dec2(neighb, feats)
        # nb_cluster2 = nb_neighs2 * nb_cluster1
        # decoded2 = nb_cluster2 x 3
        # feats2 = nb_cluster2 x 3

        decoded3 = self.dec3(feats2)
        # nb_cluster3 = nb_neighs3 * nb_cluster2
        # decoded3 = nb_cluster3 x 3

        return decoded, decoded2, decoded3

