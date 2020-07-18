import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class GridDeformationDecoder(nn.Module):
    def __init__(self, input_size, nbs_features_global, nbs_features, nb_neighbors=25, leaky=False):
        super(GridDeformationDecoder, self).__init__()
        assert(len(nbs_features) > 0)

        self.fc_layers_global = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        self.input_size = input_size
        self.nb_neighbors = nb_neighbors
        self.device = torch.device("cpu")
        self.grid = GridDeformationDecoder.create_grid(
            self.nb_neighbors, self.device
        )

        self.leaky = leaky

        self.initiate_fc_layers(nbs_features_global, nbs_features)

    def initiate_fc_layers(self, nbs_features_global, nbs_features):
        input_size = self.input_size
        for nb in nbs_features_global:
            self.fc_layers_global.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        input_size += 2
        for nb in nbs_features:
            self.fc_layers.append(
                nn.Linear(input_size, nb)
            )
            input_size = nb

        assert(input_size == 3)

    def forward(self, features):
        if not self.leaky:
            activation_fn = F.relu
        else:
            activation_fn = nn.LeakyReLU()

        nb_clusters = features.size(0)

        decoded_global = features
        for fc_layer in self.fc_layers_global:
            decoded_global = activation_fn(fc_layer(decoded_global))

        decoded_global_repeated = torch.repeat_interleave(
            decoded_global, self.nb_neighbors, dim=0
        )
        grid_repeated = self.grid.repeat(
            (nb_clusters, 1)
        )
        decoded = torch.cat(
            [grid_repeated, decoded_global_repeated], dim=1
        )

        i = 1
        for fc_layer in self.fc_layers:
            fc_output = fc_layer(decoded)
            if not i == len(self.fc_layers):
                decoded = activation_fn(fc_output)
            else:
                decoded = torch.tanh(fc_output)
            i += 1

        cluster = torch.arange(
            nb_clusters
        ).repeat_interleave(self.nb_neighbors)

        return decoded, cluster

    def accept_parent(self, parent):
        self.nb_neighbors = parent.get_number_neighbors()
        self.device = parent.get_device()
        self.grid = GridDeformationDecoder.create_grid(self.nb_neighbors, self.device)

    def create_grid(nb_neighbors, device):
        nb = int(math.sqrt(nb_neighbors))
        dist = 1/nb
        grid = []
        for i in range(nb):
            for j in range(nb):
                grid.append(
                    dist * torch.tensor(
                        [i + 0.5, j + 0.5]
                    )
                )
        return torch.stack(grid).to(device)

    def get_input_size(self):
        """
        :return: The input size this module expects to receive.
        """
        return self.input_size

