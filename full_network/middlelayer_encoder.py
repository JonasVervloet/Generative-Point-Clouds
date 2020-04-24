import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MiddleLayerEncoder(nn.Module):
    def __init__(self, neighborhood_enc, input_size, nbs_features, nbs_features_global, mean=False):
        super(MiddleLayerEncoder, self).__init__()

        self.neighborhood_enc = neighborhood_enc
        self.fc_layers = nn.ModuleList()
        self.fc_layers_global = nn.ModuleList()

        self.input_size = input_size
        self.output_size = nbs_features[-1]
        self.mean = mean

        self.initiate_fc_layers(input_size, nbs_features, nbs_features_global)

    def initiate_fc_layers(self, input_size, nbs_features, nbs_features_global):
        output_size = input_size + self.neighborhood_enc.get_output_size() + 3

        for nb in nbs_features:
            self.fc_layers.append(
                nn.Linear(output_size, nb)
            )
            output_size = nb

        for nb in nbs_features_global:
            self.fc_layers_global.append(
                nn.Linear(output_size, nb)
            )
            output_size = nb

        self.output_size = output_size

    def forward(self, relative_points, features, cluster):
        neighborhoods_encoding = self.neighborhood_enc(relative_points, cluster)
        neighborhoods_duplication = neighborhoods_encoding[cluster]

        encoded = torch.cat([relative_points, features, neighborhoods_duplication], 1)
        for fc_layer in self.fc_layers:
            encoded = F.relu(fc_layer(encoded))

        if self.mean:
            encoded_global = gnn.global_mean_pool(encoded, cluster)
        else:
            encoded_global = gnn.global_max_pool(encoded, cluster)

        for fc_layer in self.fc_layers_global:
            encoded_global = F.relu(fc_layer(encoded_global))

        return encoded_global
