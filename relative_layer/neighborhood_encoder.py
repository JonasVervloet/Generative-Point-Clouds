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
    def __init__(self, nbs_features, pool_index, mean=False):
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
        assert(pool_index <= len(nbs_features))

        self.fc_layers = nn.ModuleList()
        self.pool_index = pool_index
        self.output_size = 3
        self.mean = mean

        self.initiate_fc_layers(nbs_features)

    def initiate_fc_layers(self, nbs_features):
        output_size = 3
        for i in range(len(nbs_features)):
            self.fc_layers.append(
                nn.Linear(output_size,
                          nbs_features[i])
            )
            output_size = nbs_features[i]

        self.output_size = output_size

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

        encoded = points
        i = 0
        for fc_layer in self.fc_layers:
            if i == self.pool_index:
                if self.mean:
                    encoded = gnn.global_mean_pool(encoded, cluster)
                else:
                    encoded = gnn.global_max_pool(encoded, cluster)

            encoded = F.relu(fc_layer(encoded))
            i += 1

        if i == self.pool_index:
            if self.mean:
                encoded = gnn.global_mean_pool(encoded, cluster)
            else:
                encoded = gnn.global_max_pool(encoded, cluster)

        return encoded

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

