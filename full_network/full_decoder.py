import torch
import torch.nn as nn

from relative_layer.decoder import SimpleRelativeDecoder
from composed_layer.decoder import MiddleLayerDecoder


class FullDecoder(nn.Module):
    def __init__(self, nb_feats1=5, nb_feats2=25, nb_feats3=45,
                 nb_neighs1=5, nb_neighs2=20, nb_neighs3=20):
        super(FullDecoder, self).__init__()

        self.dec1 = MiddleLayerDecoder(
            nb_feats_in=nb_feats3,
            nb_feats_out=nb_feats1+nb_feats2,
            nb_outputs=nb_neighs1
        )
        self.dec2 = MiddleLayerDecoder(
            nb_feats_in=nb_feats2,
            nb_feats_out=nb_feats1,
            nb_outputs=nb_neighs2
        )
        self.dec3 = SimpleRelativeDecoder(
            nb_feats1,
            10,
            20,
            nb_neighs3
        )

        self.feats1 = nb_feats1

    def forward(self, latent):
        print("Full Network decoder")

        # latent = nb_batch x (nb_feats1 + nb_feats3)
        print(latent.size())

        latent_neighb = latent[:, :self.feats1]
        latent_feats = latent[:, self.feats1:]
        # latent_neighb = nb_batch x nb_feats1
        # latent_feats = nb_batch x nb_feats3
        print(latent_neighb.size())
        print(latent_feats.size())

        decoded, concat = self.dec1(latent_neighb, latent_feats)
        # nb_cluster = nb_neighs1 * nb_batch
        # decoded = nb_cluster x 3
        # concat = nb_cluster x (nb_feats1 + nb_feats2)
        print(decoded.size())
        print(concat.size())

        neighb = concat[:, :self.feats1]
        feats = concat[:, self.feats1:]
        # neighb = nb_cluster x nb_feats1
        # feats = nb_cluster x nb_feats2
        print(neighb.size())
        print(feats.size())

        decoded2, feats2 = self.dec2(neighb, feats)
        # nb_cluster2 = nb_neighs2 * nb_cluster1
        # decoded2 = nb_cluster2 x 3
        # feats2 = nb_cluster2 x 3
        print(decoded2.size())
        print(feats2.size())

        decoded3 = self.dec3(feats2)
        # nb_cluster3 = nb_neighs3 * nb_cluster2
        # decoded3 = nb_cluster3 x 3
        print(decoded3.size())

        return decoded, decoded2, decoded3

