import torch.nn as nn
from relative_layer.encoder import SimpleRelativeEncoder
from composed_layer.encoder import MiddleLayerEncoder


class ComposedEncoder(nn.Module):
    def __init__(self):
        super(ComposedEncoder, self).__init__()

        self.enc1 = SimpleRelativeEncoder(20, 10, 5)
        self.enc2 = MiddleLayerEncoder(25)

    def forward(self, relative_neighs, cluster, relative_neighs2, indices2, cluster2):

        # relative_neighs = (nb_cluster * nb_points) x 3
        # cluster = (nb_cluster * nb_points) x 1
        # relative_neighs2 = (nb_cluster2 * nb_points2) x 3
        # indices2 = (nb_cluster2 * nb_points2) x 1
        # cluster2 = (nb_cluster * nb_points) x 1

        feats1 = self.enc1(relative_neighs, cluster)
        # feats1 = nb_cluster x 5

        feats1_mapped = feats1[indices2]
        # feats1_mapped = (nb_cluster2 * nb_points2) x 5

        encoding, feats2 = self.enc2(relative_neighs2, feats1_mapped, cluster2)
        # encoding = nb_cluster2 x 5
        # feats2 = nb_cluster2 x 25

        return encoding, feats2