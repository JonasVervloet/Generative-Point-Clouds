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

        self.nbs_features_global = nbs_features_global
        self.nbs_features = nbs_features

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

    def to_string(self):
        string = str(MiddleLayerDecoder.__name__) + "\n"
        string += "> Input size: " + str(self.input_size) + "\n"
        string += "> Numbers of global features: " + str(self.nbs_features_global) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Leaky: " + str(self.leaky) + "\n"
        string += self.neighborhood_dec.to_string()
        return string

    @staticmethod
    def from_string(input_string_lines, reader):
        assert(input_string_lines[0] == str(MiddleLayerDecoder.__name__))

        input_size = int(input_string_lines[1].replace("> Input size: ", ""))
        nbs_features_global = list(map(
            int, input_string_lines[2].replace(
                "> Numbers of global features: [", ""
            ).replace("]", "").split(",")
        ))
        nbs_features = list(map(
            int, input_string_lines[3].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        leaky = input_string_lines[4].replace("> Leaky: ", "") == "True"
        neighborhood_dec = reader.read_network_lines(input_string_lines[5:])

        return MiddleLayerDecoder(neighborhood_dec, input_size, nbs_features_global,
                                  nbs_features, leaky)


class MiddleLayerDecoderSplit(nn.Module):
    def __init__(self, neighborhood_dec, input_size, neigh_dec_size, nbs_features, leaky=False):
        super(MiddleLayerDecoderSplit, self).__init__()
        assert(neighborhood_dec.get_input_size() == neigh_dec_size)

        self.neighborhood_dec = neighborhood_dec
        self.fc_layers = nn.ModuleList()
        self.input_size = input_size
        self.neigh_dec_size = neigh_dec_size
        self.output_size = input_size

        self.leaky = leaky
        self.nbs_features = nbs_features

        self.initialize_fc_layers(nbs_features)

    def initialize_fc_layers(self, nbs_features):
        nbs_features_in = self.input_size  + 3
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

        neighborhood_feats = input_features[:, :self.neigh_dec_size]
        features = input_features[:, self.neigh_dec_size:]

        relative_points, cluster = self.neighborhood_dec(neighborhood_feats)
        features_duplicated = features[cluster]

        decoded_features = torch.cat([features_duplicated, relative_points], dim=1)
        for fc_layer in self.fc_layers:
            decoded_features = activation_fn(fc_layer(decoded_features))

        return relative_points, decoded_features, cluster

    def to_string(self):
        string = str(MiddleLayerDecoderSplit.__name__) + "\n"
        string += "> Input size: " + str(self.input_size) + "\n"
        string += "> Neighborhood decoder size: " + str(self.neigh_dec_size) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Leaky: " + str(self.leaky) + "\n"
        string += self.neighborhood_dec.to_string()

        return string

    @staticmethod
    def from_string(input_string_lines, reader):
        assert(input_string_lines[0] == str(MiddleLayerDecoderSplit.__name__))
        print(input_string_lines)

        input_size = int(input_string_lines[1].replace("> Input size: ", ""))
        neigh_dec_size = int(input_string_lines[2].replace("> Neighborhood decoder size: ", ""))
        nbs_features = list(map(
            int, input_string_lines[3].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        leaky = input_string_lines[4].replace("> Leaky: ", "") == "True"
        neighborhood_dec = reader.read_network_lines(input_string_lines[5:])

        return MiddleLayerDecoderSplit(neighborhood_dec, input_size, neigh_dec_size,
                                       nbs_features, leaky)




