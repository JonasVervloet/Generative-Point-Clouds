import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from relative_layer.encoder import SimpleRelativeEncoder


class ComposeLayerEncoder(nn.Module):
    def __init__(self, nb_feats,  mean=False):
        super(ComposeLayerEncoder, self).__init__()
        self.encoder = SimpleRelativeEncoder(20, 10, 5)
        self.conv = nn.Conv1d(13, nb_feats, 1)
        self.fc = nn.Linear(nb_feats, nb_feats)

        self.mean = mean

    def forward(self, relative_points, features, cluster):
        print("encoder!")

        # relative_points = (nb_cluster x nb_points) x 3
        # features = (nb_cluster x nb_points) x 5
        # cluster = (nb_cluster x nb_points) x 1

        encoded = self.encoder(relative_points, cluster)
        # encoded = nb_cluster x 5
        print(encoded.size())

        encoded_mapped = encoded[cluster]
        # encoded = (nb_cluster x nb_points) x 5
        print(encoded_mapped.size())

        concat = torch.cat([relative_points, features, encoded_mapped], 1)
        # encoded = (nb_cluster x nb_points) x 13
        print(concat.size())

        unsqueezed = concat.unsqueeze(0)
        # unsqueezed = 1 x (nb_cluster x nb_points) x 13
        print(unsqueezed.size())

        transpose = unsqueezed.transpose(2, 1)
        # unsqueezed = 1 x 13 x (nb_cluster x nb_points)
        print(transpose.size())

        conv = F.relu(self.conv(transpose))
        # conv = 1 x nb_feats x (nb_cluster x nb_points)
        print(conv.size())

        transpose2 = conv.transpose(1, 2)
        # transpose2 = 1 x (nb_cluster x nb_points) x nb_feats
        print(transpose2.size())

        if self.mean:
            feats = gnn.global_mean_pool(transpose2[0], cluster)
        else:
            feats = gnn.global_max_pool(transpose2[0], cluster)
        # feats = nb_cluster x nb_feats
        print(feats.size())

        out = F.relu(self.fc(feats))
        # out = nb_cluster x nb_feats
        print(out.size())

        return out






