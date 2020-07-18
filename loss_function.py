import torch
from torch.nn import Module
import numpy as np


class ChamferDistLoss(Module):
    def __init__(self):
        super(ChamferDistLoss, self).__init__()

    def forward(self, input, output, batch_in=None, batch_out=None):
        if len(input.size()) == 3 and len(output.size()) == 3:
            loss = 0
            for i in range(input.size(0)):
                loss += self.chamfer_dist_tensor(input[i], output[i])
            return loss
        elif batch_in is None and batch_out is None:
            return self.chamfer_dist_tensor(input, output)
        elif batch_out is None:
            assert(torch.max(batch_in) + 1 == output.size(0))

            loss = 0
            for i in range(output.size(0)):
                loss += self.chamfer_dist_tensor(
                    input[batch_in==i], output[i]
                )
            return loss
        else:
            assert (batch_in is not None)
            assert (batch_out is not None)
            assert(torch.max(batch_in) == torch.max(batch_out))
            nb_batch = torch.max(batch_in)

            loss = 0
            for i in range(nb_batch):
                loss += self.chamfer_dist_tensor(
                    input[batch_in == i], output[batch_out == i]
                )
            return loss

    def chamfer_dist(self, cloud1, cloud2):
        with torch.no_grad():
            distances = self.get_distances(cloud1, cloud2)
            points = self.get_closest_points(cloud2, distances)
        loss = sum(sum((cloud1 - points)**2))
        with torch.no_grad():
            distances = distances.transpose()
            points = self.get_closest_points(cloud1, distances)
        loss += sum(sum((cloud2 - points)**2))
        return loss

    def get_closest_points(self, cloud2, distances):
        """
        Return for each point in cloud 1 the nearest point in cloud2.

        :param cloud2: A second point cloud
        :param distances: distance[i, j] equals the distance between
                            cloud1[i] and cloud2[i]
        :return: For each point in cloud1, the nearest point in cloud 2 is returned.
        """
        indices = np.argmin(distances, axis=1)
        points = cloud2[indices]
        return points

    def get_distances(self, cloud1, cloud2):
        """
        Returns a matrix with the distances between the points of both point clouds.

        :param cloud1: A point cloud
        :param cloud2: A second point cloud
        :return: result[i, j] is the distance between
                    the ith point of cloud1 and the jth point of cloud2.
        """
        return np.array(
                [np.array(
                    [sum((p - m)**2) for p in cloud2]
                ) for m in cloud1]
            )

    def get_distances_tensor(self, cloud1, cloud2):
        distances = []
        for i in range(len(cloud2)):
            dists = (cloud1 - cloud2[i]) ** 2
            distances.append(torch.sum(dists, dim=1))
        return torch.stack(distances)

    def get_closest_points_tensor(self, cloud2, distances):
        indices = torch.argmin(distances, dim=0)
        return indices

    def chamfer_dist_tensor(self, cloud1, cloud2):
        with torch.no_grad():
            distances = self.get_distances_tensor(cloud1, cloud2)
            indices = self.get_closest_points_tensor(cloud2, distances)
        points = cloud2[indices]
        loss = torch.sum((cloud1 - points) ** 2)
        with torch.no_grad():
            distances = torch.transpose(distances, 0, 1)
            indices = self.get_closest_points_tensor(cloud1, distances)
        points = cloud1[indices]
        loss += torch.sum((cloud2 - points) ** 2)
        return loss

    def to_string(self):
        string = str(ChamferDistLoss.__name__) + "\n"
        return string

    @staticmethod
    def from_string(input_string_lines):
        assert(input_string_lines[0] == str(ChamferDistLoss.__name__))
        assert(len(input_string_lines) == 1)

        return ChamferDistLoss()


class ChamferDistLossFullNetwork(Module):
    def __init__(self):
        super(ChamferDistLossFullNetwork, self).__init__()

    def forward(self, parameter_list):
        input_points = parameter_list[0][0]
        input_clusters = parameter_list[1][0]
        output_points = parameter_list[2][0]
        output_clusters = parameter_list[3][0]
        assert(torch.max(input_clusters) == torch.max(output_clusters))
        nb_clusters = torch.max(input_clusters)

        loss = 0
        for i in range(nb_clusters):
            loss += self.chamfer_dist_tensor(
                input_points[input_clusters == i], output_points[output_clusters == i]
            )
        return loss

    def chamfer_dist(self, cloud1, cloud2):
        with torch.no_grad():
            distances = self.get_distances(cloud1, cloud2)
            points = self.get_closest_points(cloud2, distances)
        loss = sum(sum((cloud1 - points)**2))
        with torch.no_grad():
            distances = distances.transpose()
            points = self.get_closest_points(cloud1, distances)
        loss += sum(sum((cloud2 - points)**2))
        return loss

    def get_closest_points(self, cloud2, distances):
        """
        Return for each point in cloud 1 the nearest point in cloud2.

        :param cloud2: A second point cloud
        :param distances: distance[i, j] equals the distance between
                            cloud1[i] and cloud2[i]
        :return: For each point in cloud1, the nearest point in cloud 2 is returned.
        """
        indices = np.argmin(distances, axis=1)
        points = cloud2[indices]
        return points

    def get_distances(self, cloud1, cloud2):
        """
        Returns a matrix with the distances between the points of both point clouds.

        :param cloud1: A point cloud
        :param cloud2: A second point cloud
        :return: result[i, j] is the distance between
                    the ith point of cloud1 and the jth point of cloud2.
        """
        return np.array(
                [np.array(
                    [sum((p - m)**2) for p in cloud2]
                ) for m in cloud1]
            )

    def get_distances_tensor(self, cloud1, cloud2):
        distances = []
        for i in range(len(cloud2)):
            dists = (cloud1 - cloud2[i]) ** 2
            distances.append(torch.sum(dists, dim=1))
        return torch.stack(distances)

    def get_closest_points_tensor(self, cloud2, distances):
        indices = torch.argmin(distances, dim=0)
        return indices

    def chamfer_dist_tensor(self, cloud1, cloud2):
        with torch.no_grad():
            distances = self.get_distances_tensor(cloud1, cloud2)
            indices = self.get_closest_points_tensor(cloud2, distances)
        points = cloud2[indices]
        loss = torch.sum((cloud1 - points) ** 2)
        with torch.no_grad():
            distances = torch.transpose(distances, 0, 1)
            indices = self.get_closest_points_tensor(cloud1, distances)
        points = cloud1[indices]
        loss += torch.sum((cloud2 - points) ** 2)
        return loss

    def to_string(self):
        string = str(ChamferDistLossFullNetwork.__name__) + "\n"
        return string

    @staticmethod
    def from_string(input_string_lines):
        assert(input_string_lines[0] == str(ChamferDistLossFullNetwork.__name__))
        assert(len(input_string_lines) == 1)

        return ChamferDistLossFullNetwork()


class ChamferVAELoss(Module):
    def __init__(self, alfa=0.5):
        super(ChamferVAELoss, self).__init__()
        self.alfa = alfa
        self.chamfer = ChamferDistLoss()

    def forward(self, inp, outp, z_mu, z_var):
        chamf_loss = self.chamfer(inp, outp)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        return chamf_loss + self.alfa * kl_loss

    def to_string(self):
        string = str(ChamferVAELoss.__name__) + "\n"
        return string

    @staticmethod
    def from_string(input_string_lines):
        assert(input_string_lines[0] == str(ChamferVAELoss.__name__))
        assert(len(input_string_lines) == 1)

        return ChamferVAELoss()


class LayerChamferDistLoss(Module):
    def __init__(self, coefficient_list):
        super(LayerChamferDistLoss, self).__init__()

        self.coefficients = coefficient_list
        self.chamfer = ChamferDistLoss()

    def forward(self, points_list, points_list_out, batch_list, batch_list_out):
        length = len(self.coefficients)
        assert(length == len(points_list))
        assert(length == len(batch_list))
        assert(length == len(points_list_out))
        assert(length == len(batch_list_out))

        loss = 0
        for i in range(length):
            loss += self.coefficients[i] * self.chamfer(
                points_list[i],
                points_list_out[-i],
                batch_list[i],
                batch_list_out[-i]
            )

        return loss

    def to_string(self):
        string = str(LayerChamferDistLoss.__name__) + "\n"
        string += "> coefficients: " + str(self.coefficients) + "\n"

        return string

    @staticmethod
    def from_string(input_string_lines):
        assert(input_string_lines[0] == str(LayerChamferDistLoss.__name__))
        assert(len(input_string_lines) == 2)

        coefficients = list(map(
            float, input_string_lines[1].replace(
                "> coefficients: [", ""
            ).replace("]", "").split(",")
        ))

        return LayerChamferDistLoss(coefficients)


