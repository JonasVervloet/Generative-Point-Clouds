import torch
from torch.nn import Module
import numpy as np


class ChamferDistLoss(Module):
    def __init__(self):
        super(ChamferDistLoss, self).__init__()

    def forward(self, input, output):
        if len(input.size()) == 3:
            return sum([self.chamfer_dist(input[i], output[i]) for i in range(input.size(0))])
        else:
            return self.chamfer_dist(input, output)

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

