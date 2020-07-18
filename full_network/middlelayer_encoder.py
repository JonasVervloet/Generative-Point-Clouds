import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MiddleLayerEncoder(nn.Module):
    def __init__(self, neighborhood_enc, input_size, nbs_features, nbs_features_global, mean=False, leaky=False):
        super(MiddleLayerEncoder, self).__init__()

        self.neighborhood_enc = neighborhood_enc
        self.fc_layers = nn.ModuleList()
        self.fc_layers_global = nn.ModuleList()

        self.input_size = input_size
        self.output_size = nbs_features[-1]
        self.nbs_features = nbs_features
        self.nbs_features_global = nbs_features_global
        self.mean = mean
        self.leaky = leaky

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
        if not self.leaky:
            activation_fn = F.relu
        else:
            activation_fn = nn.LeakyReLU

        neighborhoods_encoding = self.neighborhood_enc(relative_points, cluster)
        neighborhoods_duplication = neighborhoods_encoding[cluster]

        encoded = torch.cat([relative_points, features, neighborhoods_duplication], 1)
        for fc_layer in self.fc_layers:
            encoded = activation_fn(fc_layer(encoded))

        if self.mean:
            encoded_global = gnn.global_mean_pool(encoded, cluster)
        else:
            encoded_global = gnn.global_max_pool(encoded, cluster)

        for fc_layer in self.fc_layers_global:
            encoded_global = activation_fn(fc_layer(encoded_global))

        return encoded_global

    def to_string(self):
        string = str(MiddleLayerEncoder.__name__) + "\n"
        string += "> Input size: " + str(self.input_size) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Numbers of global features: " + str(self.nbs_features_global) + "\n"
        string += "> Mean: " + str(self.mean) + "\n"
        string += "> Leaky: " + str(self.leaky) + "\n"
        string += self.neighborhood_enc.to_string()
        return string

    @staticmethod
    def from_string(input_string_lines, reader):
        assert(input_string_lines[0] == str(MiddleLayerEncoder.__name__))

        input_size = int(input_string_lines[1].replace("> Input size: ", ""))
        nbs_features = list(map(
            int, input_string_lines[2].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        nbs_features_global = list(map(
            int, input_string_lines[3].replace(
                "> Numbers of global features: [", ""
            ).replace("]", "").split(",")
        ))
        mean = input_string_lines[4].replace("> Mean: ", "") == "True"
        leaky = input_string_lines[5].replace("> Leaky: ", "") == "True"
        neighborhood_enc = reader.read_network_lines(input_string_lines[6:])

        return MiddleLayerEncoder(neighborhood_enc, input_size, nbs_features,
                                  nbs_features_global, mean, leaky)


class MiddleLayerEncoderSplit(nn.Module):
    def __init__(self, neighborhood_enc, input_size, nbs_features, nbs_features_global, mean=False, leaky=False):
        super(MiddleLayerEncoderSplit, self).__init__()

        self.neighborhood_enc = neighborhood_enc
        self.fc_layers = nn.ModuleList()
        self.fc_layers_global = nn.ModuleList()

        self.input_size = input_size
        self.output_size = nbs_features[-1]
        self.mean = mean
        self.leaky = leaky

        self.nbs_features = nbs_features
        self.nbs_features_global = nbs_features_global

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
        if not self.leaky:
            activation_fn = F.relu
        else:
            activation_fn = nn.LeakyReLU()

        neighborhoods_encoding = self.neighborhood_enc(relative_points, cluster)
        neighborhoods_duplication = neighborhoods_encoding[cluster]

        encoded = torch.cat([relative_points, features, neighborhoods_duplication], 1)
        for fc_layer in self.fc_layers:
            encoded = activation_fn(fc_layer(encoded))

        if self.mean:
            encoded_global = gnn.global_mean_pool(encoded, cluster)
        else:
            encoded_global = gnn.global_max_pool(encoded, cluster)

        for fc_layer in self.fc_layers_global:
            encoded_global = activation_fn(fc_layer(encoded_global))

        return torch.cat([neighborhoods_encoding, encoded_global], dim=1)

    def to_string(self):
        string = str(MiddleLayerEncoderSplit.__name__) + "\n"
        string += "> Input size: " + str(self.input_size) + "\n"
        string += "> Numbers of features: " + str(self.nbs_features) + "\n"
        string += "> Numbers of global features: " + str(self.nbs_features_global) + "\n"
        string += "> Mean: " + str(self.mean) + "\n"
        string += "> Leaky: " + str(self.leaky) + "\n"
        string += self.neighborhood_enc.to_string()

        return string

    @staticmethod
    def from_string(input_string_lines, reader):
        assert(input_string_lines[0] == str(MiddleLayerEncoderSplit.__name__))
        print(input_string_lines)

        input_size = int(input_string_lines[1].replace("> Input size: ", ""))
        nbs_features = list(map(
            int, input_string_lines[2].replace(
                "> Numbers of features: [", ""
            ).replace("]", "").split(",")
        ))
        nbs_features_global = list(map(
            int, input_string_lines[3].replace(
                "> Numbers of global features: [", ""
            ).replace("]", "").split(",")
        ))
        mean = input_string_lines[4].replace("> Mean: ", "") == "True"
        leaky = input_string_lines[5].replace("> Leaky: ", "") == "True"
        neighborhood_enc = reader.read_network_lines(input_string_lines[6:])

        return MiddleLayerEncoderSplit(neighborhood_enc, input_size, nbs_features,
                                       nbs_features_global, mean, leaky)


