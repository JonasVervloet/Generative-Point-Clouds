import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class PointCloudAE(nn.Module):
    def __init__(self, nb_layers, nbs_neighbors, radii, encoders, decoders):
        super(PointCloudAE, self).__init__()

        assert(nb_layers == len(nbs_neighbors))
        assert(nb_layers == len(radii))
        assert(nb_layers == len(encoders))
        assert(nb_layers == len(decoders))

        self.nb_layers = nb_layers
        self.nbs_neighbors = nbs_neighbors
        self.radii = radii

        self.validate_encoders(encoders)
        self.validate_decoders(decoders)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

    def validate_encoders(self, encoders):
        # TODO: implement
        return

    def validate_decoders(self, decoders):
        # TODO: implement
        return

    def forward(self, batch_object):
        points = batch_object.pos
        batch = batch_object.batch
        normals = None

        # if self.normal_required:
        #     normals = batch_object.norm
        encoded, points_list, batch_list = self.encode(points, batch, normals)
        points_list_out, batch_list_out = self.decode(encoded)

        return points_list, batch_list, points_list_out, batch_list_out

    def encode(self, points, batch, normals=None):
        current_points = points
        new_points = points
        current_batch = points
        new_batch = batch
        current_features = None
        new_features = None

        points_list = []
        batch_list = []

        max_i = self.nb_layers - 1
        for i in range(self.nb_layers):
            current_points = new_points
            current_batch = new_batch
            current_features = new_features

            if i == 0 and i == max_i:
                relative_points = current_points / self.radii[i]
                clusters = current_batch

                new_features = self.encoders[i](relative_points, clusters)

            elif i == max_i:
                relative_points = current_points / self.radii[i]
                clusters = current_batch
                features = current_features

                new_features = self.encoders[i](relative_points, features, clusters)

            elif i == 0:
                ratio = 1 / self.nbs_neighbors[i]
                fps_inds = gnn.fps(current_points, batch=current_batch, ratio=ratio)

                new_points = current_points[fps_inds]
                new_batch = current_batch[fps_inds]

                radius = self.radii[i]
                rad_cluster, rad_inds = gnn.radius(
                    current_points, new_points,
                    batch_x=current_batch, batch_y=new_batch, r=radius
                )

                rad_points = current_points[rad_inds]
                mid_points = new_points[rad_cluster]
                relative_points = (rad_points - mid_points) / radius
                clusters = rad_cluster

                new_features = self.encoders[i](relative_points, clusters)

            else:
                ratio = 1 / self.nbs_neighbors[i]
                fps_inds = gnn.fps(current_points, batch=current_batch, ratio=ratio)

                new_points = current_points[fps_inds]
                new_batch = current_batch[fps_inds]

                radius = self.radii[i]
                rad_cluster, rad_inds = gnn.radius(
                    current_points, new_points,
                    batch_x=current_batch, batch_y=new_batch, r=radius
                )

                rad_points = current_points[rad_inds]
                mid_points = new_points[rad_cluster]
                relative_points = (rad_points - mid_points) / radius
                features = current_features[rad_inds]
                clusters = rad_cluster

                new_features = self.encoders[i](relative_points, features, clusters)

            points_list.append(current_points)
            batch_list.append(current_batch)

        encoded = new_features

        return encoded, points_list, batch_list

    def decode(self, encoded):
        current_features = encoded
        current_batch = torch.arange(encoded.size(0))
        current_points = None
        new_features = encoded
        new_batch = torch.arange(encoded.size(0))
        new_points = None

        points_list = []
        batch_list = []

        max_i = self.nb_layers - 1
        for i in range(self.nb_layers):
            current_features = new_features
            current_batch = new_batch
            current_points = new_points

            rad_index = -(i + 1)

            if i == 0 and i == max_i:
                points, clusters = self.decoders[i](current_features)
                new_batch = current_batch[clusters]
                new_points = points * self.radii[rad_index]

            elif i == 0:
                points, features, clusters = self.decoders[i](current_features)
                new_features = features
                new_batch = current_batch[clusters]
                new_points = points * self.radii[rad_index]

            elif i == max_i:
                points, clusters = self.decoders[i](current_features)
                new_batch = current_batch[clusters]
                mid_points = current_points[clusters]
                new_points = mid_points + (points * self.radii[rad_index])

            else:
                points, features, clusters = self.decoders[i](current_features)
                new_features = features
                new_batch = current_batch[clusters]
                mid_points = current_points[clusters]
                new_points = mid_points + (points * self.radii[rad_index])

            points_list.append(new_points)
            batch_list.append(new_batch)

        return points_list, batch_list



