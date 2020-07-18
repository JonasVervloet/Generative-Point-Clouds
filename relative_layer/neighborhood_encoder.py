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
    def __init__(self, nbs_features, nbs_features_global, mean=False, leaky=False):
        """
        Initializes the parameters of this module.
        :param nbs_features: A list of three integers representing the number of
            features at each stage of the encoding. These numbers are translated
            into three different fully connected networks. The first fc network
            works on each input point separately after which a clusterwise pooling
            is performed. The other two networks work on these pooled vectors.
        :param mean: A boolean representing whether or not the pool operation should
            use a mean operation or the default max operation. Both operations are
            size and permutation invariant.
        """
        super(NeighborhoodEncoder, self).__init__()
        assert(len(nbs_features) > 0)

        self.fc_layers = nn.ModuleList()
        self.fc_layers_global = nn.ModuleList()
        self.output_size = 3
        self.nbs_features = nbs_features
        self.nbs_features_global = nbs_features_global
        self.mean = mean
        self.leaky = leaky

        self.initiate_fc_layers(nbs_features, nbs_features_global)

    def initiate_fc_layers(self, nbs_features, nbs_features_global):
        input_size = 3
        for nb in nbs_features:
            self.fc_layers.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        for nb in nbs_features_global:
            self.fc_layers_global.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        self.output_size = input_size

    def forward(self, points, cluster):
        """
        The train function of this network.
        :param points: The input points. A torch tensor of dimension
            [number of clusters * number of points in each cluster, 3].
        :param cluster: Torch tensor that assigns each point in points to
            a certain cluster.
        :return: Returns for each cluster a single encoded vector of size
            [number of clusters, output_size].
        """
        if not self.leaky:
            activation_fn = F.relu
        else:
            activation_fn = nn.LeakyReLU

        encoded = points
        for fc_layer in self.fc_layers:
            encoded = activation_fn(fc_layer(encoded))

        if self.mean:
            encoded = gnn.global_mean_pool(encoded, cluster)
        else:
            encoded = gnn.global_max_pool(encoded, cluster)

        encoded_global = encoded
        for fc_layer in self.fc_layers_global:
            encoded_global = activation_fn(fc_layer(encoded_global))

        return encoded_global

    def get_output_size(self):
        return self.output_size

    @staticmethod
    def accept_parent(parent):
        """
        Accepts a parent network. Used to communicate to the parent that this
            encoder does not need normal information.
        :param parent: A parent module that uses this module as part of its network.
        """
        parent.set_normal_not_required()

    def to_string(self):
        string = str(NeighborhoodEncoder.__name__) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Numbers of global features: " + str(self.nbs_features_global) + "\n"
        string += "> Mean: " + str(self.mean) + "\n"
        string += "> Leaky: " + str(self.leaky) + "\n"
        return string

    @staticmethod
    def from_string(input_string_list, reader):
        assert(len(input_string_list) == 5)
        assert(input_string_list[0] == str(NeighborhoodEncoder.__name__))

        nbs_features = list(map(
            int, input_string_list[1].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        nbs_global_features = list(map(
            int, input_string_list[2].replace(
                "> Numbers of global features: [", ""
            ).replace("]", "").split(",")
        ))
        mean = input_string_list[3].replace("> Mean: ", "") == "True"
        leaky = input_string_list[4].replace("> Leaky: ", "") == "True"

        return NeighborhoodEncoder(
            nbs_features, nbs_global_features, mean, leaky
        )


