import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from pointnet.pointnet import PointNet


class PointNetPP(nn.Module):
    def __init__(self, ratio1=0.25, ratio2=0.25, k=4):
        super(PointNetPP, self).__init__()
        self.pn1 = PointNet(3, 8, 16, 16)
        self.pn2 = PointNet(16, 32, 64, 64)
        self.pn3 = PointNet(64, 128, 256, 256)
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.k = k

    def forward(self, points, batch):

        # points = (nb_batch * nb_points) x 3
        # batch = (nb_batch * nb_points)
        n_points1, n_feats1, n_batch1 = self.apply_point_net(
            points, points, batch, self.pn1, self.ratio1
        )
        # nb_points1 = ration1 * nb_points
        # n_points1 = (nb_batch * nb_points1) x 3
        # n_feats1 = (nb_batch * nb_points1) x 64
        # n_batch1 = (nb_batch * nb_points1)
        n_points2, n_feats2, n_batch2 = self.apply_point_net(
            n_points1, n_feats1, n_batch1, self.pn2, self.ratio2
        )
        # nb_points2 = ration2 * nb_points
        # n_points2 = (nb_batch * nb_points2) x 3
        # n_feats2 = (nb_batch * nb_points2) x 256
        # n_batch2 = (nb_batch * nb_points2)
        n_feats3, n_batch3 = self.pn3(n_feats2, n_batch2, n_batch2)
        # n_feats3 = nb_batch x 1024
        # n_batch2 = nb_batch

        return n_feats3

    def apply_point_net(self, points, feats, batch, point_net, ratio):

        # points = (nb_batch * nb_points) x 3
        # feats = (nb_batch * nb_points) x nb_feats
        # batch = (nb_batch * nb_points)
        fps_inds = gnn.fps(points, batch, ratio=ratio)
        # fps_inds = (nb_batch * nb_points * ratio)
        n_points = points[fps_inds]
        # n_points = (nb_batch * nb_points * ratio) x 3
        n_batch = batch[fps_inds]
        # n_batch = (nb_batch * nb_points * ratio)
        knn_cluster, indices = gnn.knn(points, n_points, self.k, batch, n_batch)
        # knn_cluster = (nb_batch * nb_points * ratio * k)
        # knn_indices = (nb_batch * nb_points * ratio * k)
        knn_feats = feats[indices]
        # knn_feats = (nb_batch * nb_points * ratio * k) x nb_feats
        knn_batch = batch[indices]
        # knn_batch = (nb_batch * nb_points * ratio * k)
        n_feats, nn_batch = point_net(knn_feats, knn_batch, knn_cluster)
        # n_feats = (nb_batch * nb_points * ratio) x new_nb_feats
        # nn_batch = (nb_batch * nb_points * ratio)d

        # n_batch and nn_batch should be equal, otherwise point_net mixed something up
        assert torch.all(n_batch.eq(nn_batch))

        return n_points, n_feats, n_batch
