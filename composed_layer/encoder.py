import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from relative_layer.encoder import SimpleRelativeEncoder


class MiddleLayerEncoder(nn.Module):
    def __init__(self, nb_feats_neighb=5, nb_feats_in=5, nb_feats_out=25,  mean=False):
        super(MiddleLayerEncoder, self).__init__()

        self.encoder = SimpleRelativeEncoder(
            20,
            10,
            nb_feats_out=nb_feats_neighb
        )
        self.conv = nn.Conv1d(
            in_channels=nb_feats_in + nb_feats_neighb + 3,
            out_channels=nb_feats_out,
            kernel_size=1
        )
        self.fc = nn.Linear(
            in_features=nb_feats_out,
            out_features=nb_feats_out
        )

        self.mean = mean

    def forward(self, relative_points, features, cluster):

        # relative_points = (nb_cluster x nb_points) x 3
        # features = (nb_cluster x nb_points) x 5
        # cluster = (nb_cluster x nb_points) x 1

        encoded = self.encoder(relative_points, cluster)
        # encoded = nb_cluster x 5

        encoded_mapped = encoded[cluster]
        # encoded = (nb_cluster x nb_points) x 5

        concat = torch.cat([relative_points, features, encoded_mapped], 1)
        # encoded = (nb_cluster x nb_points) x 13

        unsqueezed = concat.unsqueeze(0)
        # unsqueezed = 1 x (nb_cluster x nb_points) x 13

        transpose = unsqueezed.transpose(2, 1)
        # unsqueezed = 1 x 13 x (nb_cluster x nb_points)

        conv = F.relu(self.conv(transpose))
        # conv = 1 x nb_feats x (nb_cluster x nb_points)

        transpose2 = conv.transpose(1, 2)
        # transpose2 = 1 x (nb_cluster x nb_points) x nb_feats

        if self.mean:
            feats = gnn.global_mean_pool(transpose2[0], cluster)
        else:
            feats = gnn.global_max_pool(transpose2[0], cluster)
        # feats = nb_cluster x nb_feats

        feats_out = F.relu(self.fc(feats))
        # feats2 = nb_cluster x nb_feats

        return encoded, feats_out






