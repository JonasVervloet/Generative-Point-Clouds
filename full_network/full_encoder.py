import torch
import torch.nn as nn

from relative_layer.encoder import SimpleRelativeEncoder
from composed_layer.encoder import MiddleLayerEncoder


class FullEncoder(nn.Module):
    def __init__(self, nb_feats1=5, nb_feats2=25, nb_feats3=45):
        super(FullEncoder, self).__init__()

        self.enc1 = SimpleRelativeEncoder(
            20,
            10,
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
        print("Full Network encoder")

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

        print(relatives.size())
        print(cluster.size())

        print(relatives2.size())
        print(indices2.size())
        print(cluster2.size())

        print(relatives3.size())
        print(cluster3.size())

        feats1 = self.enc1(relatives, cluster)
        # feats1 = nb_cluster x nb_feats1
        print(feats1.size())

        feats1_mapped = feats1[indices2]
        # feats1_mapped = (nb_cluster2 * nb_points2) x nb_feats1
        print(feats1_mapped.size())

        encoding, feats = self.enc2(relatives2, feats1_mapped, cluster2)
        # encoding = nb_cluster2 x nb_feats1
        # feats = nb_cluster2 x nb_feats2
        print(encoding.size())
        print(feats.size())

        concat = torch.cat([encoding, feats], dim=1)
        # concat = nb_cluster2 x (nb_feats1 + nb_feats2)
        print(concat.size())

        encoding2, feats2 = self.enc3(relatives3, concat, cluster3)
        # encoding2 = nb_cluster3 x nb_feats1
        # feats3 = nb_cluster3 x nb_feats3
        print(encoding2.size())
        print(feats2.size())

        concat2 = torch.cat([encoding2, feats2], dim=1)
        # concat2 = nb_cluster3 x (nb_feats1 + nb_feats3
        print(concat2.size())

        return concat2

