import torch
import torch.nn as nn

from relative_layer.encoder import SimpleRelativeEncoder
from composed_layer.encoder import MiddleLayerEncoder


class FullEncoder(nn.Module):
    def __init__(self, nb_feats1=20, nb_feats2=30, nb_feats3=80):
        super(FullEncoder, self).__init__()

        self.enc1 = SimpleRelativeEncoder(
            80,
            40,
            nb_feats_out=nb_feats1
        )
        self.enc2 = MiddleLayerEncoder(
            nb_feats_neighb=nb_feats1,
            nb_feats_in=nb_feats1,
            nb_feats_out=nb_feats2
        )
        self.enc3 = MiddleLayerEncoder(
            nb_feats_neighb=nb_feats1,
            nb_feats_in=nb_feats1 + nb_feats2,
            nb_feats_out=nb_feats3
        )

    def forward(self, relatives, cluster, relatives2, indices2, cluster2, relatives3, cluster3):

        # nb_points: nb of points within a certain radius of the points in relatives
        # relatives = (nb_cluster * nb_points) x 3
        # cluster = (nb_cluster * nb_points) x 1

        # nb_points2: nb of points within a certain radius of the points in relatives2
        # relatives2 = (nb_cluster2 * nb_points2) x 3
        # indices2 = (nb_cluster2 * nb_points2) x 1
        # cluster2 = (nb_cluster2 * nb_points2) x 1

        # nb_points3: nb of points within a certain radius of the points in relative 3
        # relatives3 = (nb_cluster3 * nb_points3) x 3
        # cluster3 = (nb_cluster3 * nb_points3) x 1

        feats1 = self.enc1(relatives, cluster)
        # feats1 = nb_cluster x nb_feats1

        feats1_mapped = feats1[indices2]
        # feats1_mapped = (nb_cluster2 * nb_points2) x nb_feats1

        encoding, feats = self.enc2(relatives2, feats1_mapped, cluster2)
        # encoding = nb_cluster2 x nb_feats1
        # feats = nb_cluster2 x nb_feats2

        concat = torch.cat([encoding, feats], dim=1)
        # concat = nb_cluster2 x (nb_feats1 + nb_feats2)

        encoding2, feats2 = self.enc3(relatives3, concat, cluster3)
        # encoding2 = nb_cluster3 x nb_feats1
        # feats3 = nb_cluster3 x nb_feats3

        concat2 = torch.cat([encoding2, feats2], dim=1)
        # concat2 = nb_cluster3 x (nb_feats1 + nb_feats3

        return concat2

