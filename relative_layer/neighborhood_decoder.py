import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodDecoder(nn.Module):
    """
    Module that decodes a single vector into a local neighborhood of 3D points.
    """
    def __init__(self, input_size, nbs_features_global, nbs_features, nb_neighbors=25):
        """
        Initializes the parameters of this module.
        :param nbs_features: A list of three integers representing the number of
            features at each stage of the decoding. These numbers are translated
            into four different fully connected networks. The first three networks
            work on the encoded input vector. The last network works on each point
            in the neighborhood separately.
        :param nb_neighbors: The number of neighbors in the local neighborhood of each
            cluster. This represents the number of points that will be decoded for each
            cluster.
        """
        super(NeighborhoodDecoder, self).__init__()

        self.fc_layers_global = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.unpool_layer = None

        self.input_size = input_size
        self.middle_size = None
        self.nb_neighbors = nb_neighbors

        self.nbs_features_global = nbs_features_global
        self.nbs_features = nbs_features

        self.initiate_fc_layers(nbs_features_global, nbs_features)
        self.create_unpool_layer()

    def initiate_fc_layers(self, nbs_features_global, nbs_features):
        input_size = self.input_size
        for nb in nbs_features_global:
            self.fc_layers_global.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        self.middle_size = input_size

        for nb in nbs_features:
            self.fc_layers.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        assert(input_size == 3)

    def create_unpool_layer(self):
        """
        This function creates a unpool layer based on the current size
            in the middle of the network and the current number of neighbors
            that should be decoded for each cluster.
        """
        print("WARNING: all training results so far will be overwritten and lost! [NeighborhoodDecoder]")
        self.unpool_layer = nn.Linear(
            self.middle_size, self.middle_size * self.nb_neighbors
        )

    def get_input_size(self):
        """
        :return: The input size this module expects to receive.
        """
        return self.input_size

    def forward(self, features):
        """
        The train function of this network.
        :param features: The input feature vectors that are decoded into
            local neighborhood points. The input is expected to be of size
            [number of clusters, input_size].
        :return: Returns a torch tensor of 3D points and a torch tensor that
            assigns each point to a cluster.
        """
        assert(features.size(1) == self.input_size)
        nb_clusters = features.size(0)

        decoded_global = features
        for fc_layer in self.fc_layers_global:
            decoded_global = F.relu(fc_layer(decoded_global))

        unpool = self.unpool_layer(decoded_global)
        decoded = unpool.view(nb_clusters * self.nb_neighbors, -1)

        for fc_layer in self.fc_layers:
            decoded = fc_layer(F.relu(decoded))
        decoded = torch.tanh(decoded)

        cluster = torch.arange(
            nb_clusters
        ).repeat_interleave(self.nb_neighbors)

        return decoded, cluster

    def accept_parent(self, parent):
        """
        Accepts a parent network. Used to sync the number of neighbors of this network with
            the number of neighbors used in the parent network.
        :param parent: A parent module that uses this module as part of its network.
        """
        self.nb_neighbors = parent.get_number_neighbors()
        self.create_unpool_layer()

    def to_string(self):
        string = str(NeighborhoodDecoder.__name__) + "\n"
        string += "> Input size: " + str(self.input_size) + "\n"
        string += "> Numbers of global features: " + str(self.nbs_features_global) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Number of neighbors: " + str(self.nb_neighbors) + "\n"
        return string

    @staticmethod
    def from_string(input_string_list, reader):
        assert(len(input_string_list) == 5)
        assert(input_string_list[0] == str(NeighborhoodDecoder.__name__))

        print(input_string_list)

        input_size = int(input_string_list[1].replace("> Input size: ", ""))
        nbs_features_global = list(map(
            int, input_string_list[2].replace(
                "> Numbers of global features: [", ""
            ).replace("]", "").split(",")
        ))
        nbs_features = list(map(
            int, input_string_list[3].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        nb_neighbors = int(input_string_list[4].replace("> Number of neighbors: ", ""))

        return NeighborhoodDecoder(
            input_size, nbs_features_global, nbs_features, nb_neighbors
        )

