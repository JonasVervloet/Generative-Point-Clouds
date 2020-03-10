import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F


class SimpleRelativeEncoder(nn.Module):
    def __init__(self, nb_feats1, nb_feats2, nb_feats_out, mean=False):
        super(SimpleRelativeEncoder, self).__init__()
        self.conv = nn.Conv1d(3, nb_feats1, 1)

        self.fc1 = nn.Linear(nb_feats1, nb_feats2)
        self.fc2 = nn.Linear(nb_feats2, nb_feats_out)

        self.mean = mean

    def forward(self, points, batch):

        # points = (nb_batch * nb_points) x 3
        # batch = (nb_batch * nb_points)

        unsqueezed = points.unsqueeze(0)
        # unsqueezed = 1 x (nb_batch * nb_points) x 3

        transpose = unsqueezed.transpose(2, 1)
        # transpose = 1 x 3 x (nb_batch * nb_points)

        conv = F.relu(self.conv(transpose))
        # conv = 1 x nb_feats1 x (nb_batch * nb_points)

        transpose2 = conv.transpose(1, 2)
        # transpose2 = 1 x (nb_batch * nb_points) x nb_feats1

        if self.mean:
            feats1 = gnn.global_mean_pool(transpose2[0], batch)
        else:
            feats1 = gnn.global_max_pool(transpose2[0], batch)
        # feats1 = nb_batch x nb_feats1

        fc1 = F.relu(self.fc1(feats1))
        # fc1 = nb_batch x nb_feats2
        
        out = F.relu(self.fc2(fc1))
        # out = nb_batch x nb_feats_out

        return out





