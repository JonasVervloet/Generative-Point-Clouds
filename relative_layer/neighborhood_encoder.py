import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodEncoder(nn.Module):
    """
    This module encodes a local neighborhood of 3D points into a single vector.
        The module is permutation invariant and can handle neighborhoods of
        different sizes.
    """
    def __init__(self, nb_features, mean=False):
        """
        Initializes the parameters of this module.
        :param nb_features: A list of three integers representing the number of
            features at each stage of the encoding. These numbers are translated
            into three different fully connected networks. The first fc network
            works on each input point separately after which a clusterwise pooling
            is performed. The other two networks work on these pooled vectors.
        :param mean: A boolean representing whether or not the pool operation should
            use a mean operation or the default max operation. Both operations are
            size and permutation invariant.
        """
        super(NeighborhoodEncoder, self).__init__()

        assert(len(nb_features) == 3)

        self.fc1 = nn.Linear(3, nb_features[0])
        self.fc2 = nn.Linear(nb_features[0], nb_features[1])
        self.fc3 = nn.Linear(nb_features[1], nb_features[2])

        self.mean = mean

    def forward(self, points, cluster):
        """
        The train function of this network.
        :param points: The input points. A torch tensor of dimension
            [number of clusters * number of points in each cluster, 3].
        :param cluster: Torch tensor that assigns each point in points to
            a certain cluster.
        :return: Returns for each cluster a single encoded vector of size
            nb_features[2].
        """
        # points = (nb_cluster * nb_points) x 3
        # batch = (nb_cluster * nb_points)

        # fc1 = (nb_cluster * nb_points) x nb_features[0]
        fc1 = F.relu(self.fc1(points))

        # pool = nb_cluster x nb_features[0]
        if self.mean:
            pool = gnn.global_mean_pool(fc1, cluster)
        else:
            pool = gnn.global_max_pool(fc1, cluster)

        # fc2 = nb_cluster x nb_features[1]
        fc2 = F.relu(self.fc2(pool))

        # out = nb_cluster x nb_features[2]
        return F.relu(self.fc3(fc2))

    @staticmethod
    def accept_parent(parent):
        """
        Accepts a parent network. Used to communicate to the parent that this
            encoder does not need normal information.
        :param parent: A parent module that uses this module as part of its network.
        """
        parent.set_normal_not_required()

