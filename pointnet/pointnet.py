import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class PointNet(nn.Module) :
    def __init__(self, nb_feats_in, nb_feat_middle1, nb_feat_middle2, nb_feats_out, final=False):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(nb_feats_in, nb_feat_middle1, 1)
        self.conv2 = nn.Conv1d(nb_feat_middle1, nb_feat_middle2, 1)
        self.conv3 = nn.Conv1d(nb_feat_middle2, nb_feats_out, 1)
        self.nb_feat_out = nb_feats_out
        self.final = final

    def forward(self, feats, batch, clusters):

        # feats = (nb_batch * nb_points) x nb_feats
        # batch = (nb_batch * nb_points)
        # cluster = (nb_batch * nb_points)
        # --> new_nb_points = nb_clusters
        unsqueezed = feats.unsqueeze(0)
        # unsqueezed = 1 x (nb_batch * nb_points) x nb_feats
        transpose = unsqueezed.transpose(2, 1)
        # transpose = 1 x nb_feats x (nb_batch * nb_points)
        conv1 = F.relu(self.conv1(transpose))
        # conv1 = 1 x nb_feats_middle1 x (nb_batch * nb_points)
        conv2 = F.relu(self.conv2(conv1))
        # conv2 = 1 x nb_feats_middle2 x (nb_batch * nb_points)
        conv3 = F.relu(self.conv3(conv2))
        # conv3 = 1 x nb_feats_out x (nb_batch * nb_points)
        transpose2 = conv3.transpose(1, 2)
        # transpose2 = 1 x (nb_batch * nb_points) x nb_feats_out
        n_feats, n_batch = gnn.max_pool_x(clusters, transpose2[0], batch)
        # n_feats = (nb_batch * new_nb_points) x nb_feats_out
        # n_batch = (nb_batch * new_nb_points)

        if self.final:
            print("sigmoid!")
            n_feats = torch.sigmoid(n_feats)

        return n_feats, n_batch

