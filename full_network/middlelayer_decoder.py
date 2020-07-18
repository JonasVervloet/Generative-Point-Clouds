import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MiddleLayerDecoder(nn.Module):
    def __init__(self, neighborhood_dec, input_size, nbs_features_global, nbs_features, leaky=False):
        super(MiddleLayerDecoder, self).__init__()
        assert(len(nbs_features_global) > 0)
        assert(neighborhood_dec.get_input_size() == nbs_features_global[-1])

        self.neighborhood_dec = neighborhood_dec
        self.fc_layers_global = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = input_size

        self.leaky = leaky

        self.initialize_fc_layers(input_size, nbs_features_global, nbs_features)

    def initialize_fc_layers(self, input_size, nbs_features_global, nbs_features):
        nb_features_in = input_size
        for nb in nbs_features_global:
            self.fc_layers_global.append(
                nn.Linear(nb_features_in, nb)
            )
            nb_features_in = nb

        nbs_features_in = input_size + nbs_features_global[-1] + 3
        for nb in nbs_features:
            self.fc_layers.append(
                nn.Linear(nbs_features_in, nb)
            )
            nbs_features_in = nb
        self.output_size = nbs_features_in

    def forward(self, input_features):
        if not self.leaky:
            activation_fn = F.relu
        else:
            activation_fn = nn.LeakyReLU()

        neighborhood_feats = input_features
        for fc_layer in self.fc_layers_global:
            neighborhood_feats = activation_fn(fc_layer(neighborhood_feats))

        relative_points, cluster = self.neighborhood_dec(neighborhood_feats)
        neighborhood_feats_duplicated = neighborhood_feats[cluster]
        input_feats_duplicated = input_features[cluster]

        decoded_features = torch.cat(
            [input_feats_duplicated, neighborhood_feats_duplicated, relative_points], 1
        )
        for fc_layer in self.fc_layers:
            decoded_features = activation_fn(fc_layer(decoded_features))

        return relative_points, decoded_features, cluster
