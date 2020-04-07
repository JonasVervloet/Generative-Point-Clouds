import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from relative_layer.encoder3 import RotationInvariantEncoder
from relative_layer.decoder2 import RelativeDecoder


class RotationInvariantLayer(nn.Module):
    def __init__(self, nb_neighbours, radius, feats1, feats2, feats_out):
        super(RotationInvariantLayer, self).__init__()

        self.nb_neighbours = nb_neighbours
        self.ratio = 1/nb_neighbours
        self.radius = radius

        self.enc = RotationInvariantEncoder(feats1, feats2, feats_out)
        self.dec = RelativeDecoder(feats_out, feats2, feats1, nb_neighbours)

    def forward(self, points, normals):
        # points = (nb_cluster * nb_neighbours) x 3
        # normals = (nb_cluster * nb_neighbours) x 3

        # fps_inds = nb_cluster x 1
        # fps_points = nb_cluster x 3
        # fps_normals = nb_cluster x 3
        fps_inds = gnn.fps(points, ratio=self.ratio)
        fps_points = points[fps_inds]
        fps_normals = normals[fps_inds]

        # rad_cluster = (nb_cluster * nb_points) x 1
        # rad_inds = (nb_cluster * nb_points) x 1
        # rad_points = (nb_cluster * nb_points) x 3
        # rad_normals = (nb_cluster * nb_points) x 3
        rad_cluster, rad_inds = gnn.radius(points, fps_points, r=self.radius)
        rad_points = points[rad_inds]
        rad_normals = normals[rad_inds]

        # midpoints = (nb_cluster * nb_points) x 3
        # midpoints_normals = (nb_cluster * nb_points) x 3
        # relative = (nb_cluster * nb_points) x 3
        midpoints = fps_points[rad_cluster]
        midpoints_normals = fps_normals[rad_cluster]
        relative = (rad_points - midpoints) / self.radius

        # inv_featurs = (nb_cluster * nb_points) x 4
        inv_features = RotationInvariantLayer.inv_features(
            relative, rad_normals, midpoints_normals
        )

        # encoding = nb_cluster x nb_feats_out
        # angles = nb_cluster x 3
        encoding, angles = self.enc(relative, inv_features, rad_cluster)

        # concat = nb_cluster x (3 + nb_feats_out)
        concat = torch.cat([angles, encoding], dim=-1)

        # decoded = (nb_cluster * nb_neighbours) x 3
        decoded = self.dec(concat)

        # cluster_range = nb_cluster x 1
        # cluster_inds = (nb_cluster * nb_neighbours) x 1
        # cluster_midpoints = (nb_cluster * nb_neighbours) x 3
        cluster_range = torch.arange(0, fps_points.size(0))
        cluster_inds = cluster_range.repeat_interleave(self.nb_neighbours)
        cluster_midpoints = fps_points[cluster_inds]

        # decoded_abs = (nb_cluster * nb_neighbours) x 3
        decoded_abs = (decoded * self.radius) + cluster_midpoints

        return rad_points, rad_cluster, decoded_abs, cluster_inds

    @staticmethod
    def inv_features(distances, normals, normals_midpoint):
        angle1 = RotationInvariantLayer.get_angle(normals_midpoint, distances)
        angle2 = RotationInvariantLayer.get_angle(normals, distances)
        angle3 = RotationInvariantLayer.get_angle(normals_midpoint, normals)
        norms = torch.norm(distances, dim=-1)

        return torch.stack([angle1, angle2, angle3, norms]).transpose(0, 1)

    @staticmethod
    def get_angle(vec1, vec2):
        assert(vec1.size() == vec2.size())

        nb_rows = vec1.size(0)
        cross = torch.norm(torch.cross(vec1, vec2, dim=-1), dim=-1)
        dot = torch.bmm(
            vec1.view(nb_rows, 1, -1), vec2.view(nb_rows, -1, 1)
        )[:, 0, :].view(-1)

        return torch.atan2(cross, dot)




