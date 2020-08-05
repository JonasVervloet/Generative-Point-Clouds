import torch
from torch.nn import Module
import numpy as np


class ChamferDistLoss(Module):
    def __init__(self):
        super(ChamferDistLoss, self).__init__()

    def forward(self, input, output, batch_in=None):
        if len(input.size()) == 3:
            loss = 0
            for i in range(input.size(0)):
                loss += self.chamfer_dist_tensor(input[i], output[i])
            return loss
        elif batch_in is None:
            return self.chamfer_dist_tensor(input, output)
        else:
            loss = 0
            assert(torch.max(batch_in) + 1 == output.size(0))
            for i in range(output.size(0)):
                loss += self.chamfer_dist_tensor(
                    input[batch_in == i], output[i]
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


class ChamferVAELoss(Module):
    def __init__(self, alfa=0.5):
        super(ChamferVAELoss, self).__init__()
        self.alfa = alfa
        self.chamfer = ChamferDistLoss()

    def forward(self, inp, outp, z_mu, z_var):
        chamf_loss = self.chamfer(inp, outp)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        return chamf_loss + self.alfa * kl_loss

