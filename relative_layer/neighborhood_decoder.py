import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodDecoder(nn.Module):
    """
    Module that decodes a single vector into a local neighborhood of 3D points.
    """
    def __init__(self, nb_features, nb_neighbors=25):
        """
        Initializes the parameters of this module.
        :param nb_features: A list of three integers representing the number of
            features at each stage of the decoding. These numbers are translated
            into four different fully connected networks. The first three networks
            work on the encoded input vector. The last network works on each point
            in the neighborhood separately.
        :param nb_neighbors: The number of neighbors in the local neighborhood of each
            cluster. This represents the number of points that will be decoded for each
            cluster.
        """
        super(NeighborhoodDecoder, self).__init__()

        self.fc1 = nn.Linear(nb_features[0], nb_features[1])
        self.fc2 = nn.Linear(nb_features[1], nb_features[2])
        self.fc3 = nn.Linear(nb_features[2], nb_features[2] * nb_neighbors)
        self.fc4 = nn.Linear(nb_features[2], 3)

        self.conv = nn.Conv1d(nb_features[2], 3, 1)

        self.nb_features = nb_features
        self.nb_neighbors = nb_neighbors

    def forward(self, features):
        """
        The train function of this network.
        :param features: The input feature vectors that are decoded into
            local neighborhood points. The input is expected to be of size
            [number of clusters, nb_features[0].
        :return: Returns a torch tensor of 3D points and a torch tensor that
            assigns each point to a cluster.
        """
        # features = nb_cluster x nb_features[0]

        # fc1 = nb_cluster x nb_features[1]
        fc1 = F.relu(self.fc1(features))

        # fc2 = nb_cluster x nb_features[2]
        fc2 = F.relu(self.fc2(fc1))

        # fc3 = nb_cluster x (nb_features[2] * nb_neighbors)
        fc3 = F.relu(self.fc3(fc2))

        # resized = (nb_cluster * nb_neighbors) x nb_features[2]
        resized = fc3.view(-1, self.nb_features[2])

        # out = (nb_cluster * nb_neighbors) x 3
        out = torch.tanh(self.fc4(resized))

        # cluster = (nb_cluster * nb_neighbors) x 1
        cluster = torch.arange(
            features.size(0)
        ).repeat_interleave(self.nb_neighbors)

        return out, cluster

    def accept_parent(self, parent):
        """
        Accepts a parent network. Used to sync the number of neighbors of this network with
            the number of neighbors used in the parent network.
        :param parent: A parent module that uses this module as part of its network.
        """
        print("WARNING: all training results so far will be overwritten and lost! [SingleLayerNetwork]")
        new_nb_neighbors = parent.get_number_neighbors()
        self.nb_neighbors = new_nb_neighbors
        self.fc3 = nn.Linear(self.nb_features[2], self.nb_features[2] * new_nb_neighbors)