import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class PointCloudAE(nn.Module):
    def __init__(self, nb_layers, nbs_neighbors, radii, encoders, decoders, complete_encoding=False):
        super(PointCloudAE, self).__init__()

        assert(nb_layers == len(nbs_neighbors))
        assert(nb_layers == len(radii))
        assert(nb_layers == len(encoders))
        assert(nb_layers == len(decoders))

        self.nb_layers = nb_layers
        self.nbs_neighbors = nbs_neighbors
        self.radii = radii
        self.complete_encoding = complete_encoding

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

        input_points, input_clusters, encoding, anchor_points = self.encode(points, batch, normals)
        output_points, output_clusters = self.decode(encoding, anchor_points)

        return [input_points, input_clusters, output_points, output_clusters, encoding, anchor_points]

    def encode(self, points, batch, normals=None):
        fps_points, fps_batches = self.do_fps_computations(
            points, batch
        )
        radius_clusters, radius_indices, radius_points = self.do_radius_computations(
            fps_points, fps_batches
        )
        relative_points = self.get_relative_points(
            fps_points, radius_points, radius_clusters
        )
        features = self.encode_features(
            relative_points, radius_clusters, radius_indices
        )

        input_points, input_clusters = self.get_used_input_points(
            fps_points, radius_indices, radius_clusters
        )

        return input_points, input_clusters, features, fps_points

    def do_fps_computations(self, points, batch):
        if self.complete_encoding:
            nb_fps_computations = self.nb_layers - 1
        else:
            nb_fps_computations = self.nb_layers

        points_list = [points]
        batch_list = [batch]
        for i in range(nb_fps_computations):
            ratio = 1 / self.nbs_neighbors[i]
            fps_indices = gnn.fps(
                x=points_list[i],
                batch=batch_list[i],
                ratio=ratio
            )
            points_list.append(
                points_list[i][fps_indices]
            )
            batch_list.append(
                batch_list[i][fps_indices]
            )

        return points_list, batch_list

    def do_radius_computations(self, fps_points, fps_batches):
        if self.complete_encoding:
            nb_radius_computations = self.nb_layers - 1
        else:
            nb_radius_computations = self.nb_layers

        cluster_list = []
        indices_list = []
        points_list = []
        for i in range(nb_radius_computations):
            radius_cluster, radius_indices = gnn.radius(
                x=fps_points[i],
                y=fps_points[i+1],
                batch_x=fps_batches[i],
                batch_y=fps_batches[i+1],
                r=self.radii[i]
            )
            cluster_list.append(radius_cluster)
            indices_list.append(radius_indices)
            points_list.append(
                fps_points[i][radius_indices]
            )

        if self.complete_encoding:
            cluster_list.append(fps_batches[-1])
            indices_list.append(
                torch.arange(fps_points[-1].size(0))
            )
            points_list.append(fps_points[-1])

        return cluster_list, indices_list, points_list

    def get_relative_points(self, fps_points, radius_points, radius_clusters):
        if self.complete_encoding:
            nb_relative_computations = self.nb_layers - 1
        else:
            nb_relative_computations = self.nb_layers

        relative_points = []
        for i in range(nb_relative_computations):
            anchor_points = fps_points[i + 1][radius_clusters[i]]
            relative_points.append(
                (radius_points[i] - anchor_points) / self.radii[i]
            )

        if self.complete_encoding:
            relative_points.append(
                fps_points[-1] / self.radii[-1]
            )

        return relative_points

    def encode_features(self, relative_points, radius_clusters, radius_indices):
        features = []
        for i in range(self.nb_layers):
            encoder = self.encoders[i]
            if i == 0:
                features.append(
                    encoder(relative_points[i], radius_clusters[i])
                )
            else:
                features.append(encoder(
                    relative_points[i],
                    features[-1][radius_indices[i]],
                    radius_clusters[i]
                ))

        return features

    def get_used_input_points(self, fps_points, indices_list, cluster_list):
        input_points = []
        input_clusters = []

        # the indices that can be used to select the right points of the fps points
        current_indices = indices_list[-1]
        # the clusters that denote to which cluster every index in current_indices belongs to
        current_clusters = cluster_list[-1]

        for i in range(self.nb_layers - 1):
            index = self.nb_layers - (i + 2)
            fps_index = index
            if self.complete_encoding:
                fps_index += 1
            input_points.append(fps_points[fps_index][current_indices])
            input_clusters.append(current_clusters)

            next_indices = []
            next_clusters = []
            for j in range(len(current_indices)):
                # mask that represents which elements of the next cluster and index list should be used
                mask = cluster_list[index] == current_indices[j]
                # the indices of the next layer that belongs to the current cluster of current index j
                indices_to_append = indices_list[index][mask]
                next_indices.append(indices_to_append)
                next_clusters.append(
                    current_clusters[j].repeat(indices_to_append.size(0))
                )

            current_indices = torch.cat(next_indices, 0)
            current_clusters = torch.cat(next_clusters, 0)

        input_points.append(fps_points[0][current_indices])
        input_clusters.append(current_clusters)
        input_points.reverse()
        input_clusters.reverse()

        return input_points, input_clusters

    def decode(self, features, fps_points):
        relative_points_list, clusters_list = self.decode_features(features)
        output_points = self.construct_output_points(
            relative_points_list, clusters_list, fps_points
        )
        output_clusters = self.construct_output_clusters(
            clusters_list, features
        )

        return output_points, output_clusters

    def decode_features(self, features):
        relative_points_list = []
        clusters_list = []

        current_features = features[-1]
        for i in range(self.nb_layers):
            if not i == self.nb_layers - 1:
                relative_points, new_features, clusters = self.decoders[i](current_features)
                current_features = new_features
            else:
                relative_points, clusters = self.decoders[i](current_features)

            relative_points_list.append(relative_points)
            clusters_list.append(clusters)

        return relative_points_list, clusters_list

    def construct_output_points(self, relative_points_list, clusters, fps_points):
        output_points = []
        if not self.complete_encoding:
            output_points.append(
                fps_points[-1]
            )
        for i in range(self.nb_layers):
            if self.complete_encoding and i == 0:
                output_points.append(
                    relative_points_list[i] * self.radii(-(i+1))
                )
            else:
                anchor_points = output_points[-1][clusters[i]]
                output_points.append(
                    anchor_points + relative_points_list[i] * self.radii[-(i+1)]
                )

        output_points.reverse()
        return output_points

    def construct_output_clusters(self, clusters, features):
        output_clusters = [
            torch.arange(features[-1].size(0))
        ]
        for i in range(self.nb_layers):
            output_clusters.append(
                output_clusters[-1][clusters[i]]
            )

        output_clusters.reverse()
        return output_clusters

    def to_string(self):
        string = str(PointCloudAE.__name__) + "\n"
        string += "> Number of layers: " + str(self.nb_layers) + "\n"
        string += "> Numbers of neighbors: " + str(self.nbs_neighbors) + "\n"
        string += "> Radii: " + str(self.radii) + "\n"
        string += "> Complete encoding: " + str(self.complete_encoding) + "\n"

        string += "ENCODERS" + "\n"
        for encoder in self.encoders:
            string += "ENCODER" + "\n"
            string += encoder.to_string()

        string += "DECODERS" + "\n"
        for decoder in self.decoders:
            string += "DECODER" + "\n"
            string += decoder.to_string()

        return string

    @staticmethod
    def from_string(input_string_lines, reader):
        assert(input_string_lines[0] == str(PointCloudAE.__name__))

        nb_layers = int(input_string_lines[1].replace("> Number of layers: ", ""))
        nbs_neighbors = list(map(
            int, input_string_lines[2].replace(
                "> Numbers of neighbors: [", ""
            ).replace("]", "").split(",")
        ))
        radii = list(map(
            float, input_string_lines[3].replace(
                "> Radii: [", ""
            ).replace("]", "").split(",")
        ))
        complete_encoding = input_string_lines[4].replace(
            "> Complete encoding: ", "") == "True"

        assert(input_string_lines[5] == "ENCODERS")
        encoders = []
        current_lines = []
        i = 7
        while input_string_lines[i] != "DECODERS":
            while (input_string_lines[i] != "ENCODER" and
                   input_string_lines[i] != "DECODERS"):
                current_lines.append(input_string_lines[i])
                i += 1

            encoders.append(reader.read_network_lines(current_lines))
            current_lines = []
            if input_string_lines[i] != "DECODERS":
                i += 1

        decoders = []
        i += 2
        while i < len(input_string_lines):
            while (i < len(input_string_lines)
                   and input_string_lines[i] != "DECODER"):
                current_lines.append(input_string_lines[i])
                i += 1

            decoders.append(reader.read_network_lines(current_lines))
            current_lines = []
            i += 1

        return PointCloudAE(nb_layers, nbs_neighbors, radii,
                            encoders, decoders, complete_encoding)



