import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class RotationInvariantEncoder(nn.Module):
    def __init__(self, feats1, feats2, feats_out):
        super(RotationInvariantEncoder, self).__init__()

        self.fc1 = nn.Linear(4, feats2)
        self.fc2 = nn.Linear(feats2, feats1)

        self.fc3 = nn.Linear(feats1, feats2)
        self.fc4 = nn.Linear(feats2, feats_out)

        self.fc5 = nn.Linear(feats_out + 3, 3)

    def forward(self, points, inv_feats, cluster):
        # points = (nb_cluster * nb_points) x 3
        # inv_feats = (nb_cluster * nb_points) x 4
        # cluster = (nb_cluster * nb_points) x 1

        # fc1 = (nb_cluster * nb_points) x nb_feats2
        fc1 = torch.relu(self.fc1(inv_feats))

        # fc2 = (nb_cluster * nb_points) x nb_feats1
        fc2 = torch.relu(self.fc2(fc1))

        # max_pool = nb_cluster x nb_feats1
        max_pool = gnn.global_max_pool(fc2, cluster)

        # fc3 = nb_cluster x  nb_feats2
        fc3 = torch.relu(self.fc3(max_pool))

        # fc4 = nb_cluster x nb_feats_out
        fc4 = torch.relu(self.fc4(fc3))

        # repeated = (nb_cluster * nb_points) x nb_feats_out
        repeated = fc4[cluster]

        # concat = (nb_cluster * nb_points) x (nb_feats_out + 3)
        concat = torch.cat([points, repeated], dim=-1)

        # fc5 = (nb_cluster * nb_points) x 3
        fc5 = torch.relu(self.fc5(concat))

        # angles = nb_cluster x 3
        angles = torch.sigmoid(gnn.global_mean_pool(fc5, cluster))

        return fc4, angles




