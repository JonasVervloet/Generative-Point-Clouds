import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F


class RelativeEncoder(nn.Module):
    def __init__(self, feats1, feats2, feats_angle, feats_out, mean=False):
        super(RelativeEncoder, self).__init__()
        self.fc1 = nn.Linear(3, feats1)

        self.fc1_angle = nn.Linear(feats1, feats_angle)
        self.fc2_angle = nn.Linear(feats_angle, 3)

        self.fc1_feat = nn.Linear(feats1, feats2)
        self.fc2_feat = nn.Linear(feats2, feats_out)

        self.mean = mean

    def forward(self, points, batch):
        # points = (nb_batch * nb_points) x 3
        # batch = (nb_batch * nb_points) x 1

        # fc1 = (nb_batch * nb_points) x nb_feats1
        fc1 = F.relu(self.fc1(points))

        # feats1 = nb_batch x nb_feats1
        if self.mean:
            feats1 = gnn.global_mean_pool(fc1, batch)
        else:
            feats1 = gnn.global_max_pool(fc1, batch)

        # fc1_angle = nb_batch x nb_feats_angle
        fc1_angle = F.relu(self.fc1_angle(feats1))

        # fc2_angle = nb_batch x 3
        fc2_angle = torch.sigmoid(self.fc2_angle(fc1_angle))

        # fc1_feat = nb_batch x nb_feats2
        fc1_feat = F.relu(self.fc1_feat(feats1))

        # fc2_feat = nb_batch x nb_feats_out
        fc2_feat = F.relu(self.fc2_feat(fc1_feat))

        # out = nb_batch x (2 + nb_feats_out)
        return torch.cat([fc2_angle, fc2_feat], dim=1)





