from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder

from full_network.middlelayer_encoder import MiddleLayerEncoder, MiddleLayerEncoderSplit
from full_network.middlelayer_decoder import MiddleLayerDecoder, MiddleLayerDecoderSplit
from full_network.point_cloud_ae import PointCloudAE


class NetworkGenerator:
    def __init__(self):
        self.nb_layers = 1
        self.nbs_neighbors = []
        self.radii = []
        self.latent_sizes = []
        self.leaky = False
        self.complete_encoding = False
        self.mean = False
        self.grid = True
        self.split = False

    """Number of layers"""
    def set_nb_layers(self, number):
        assert (number >= 1)
        assert (number <= 3)
        self.nb_layers = number

    """Numbers of neighbors"""
    def set_nbs_neighbors(self, numbers):
        self.nbs_neighbors = numbers

    def add_nb_neighbor(self, number):
        self.nbs_neighbors.append(number)

    """Numbers of radius"""
    def set_radii(self, radii):
        self.radii = radii

    def add_radius(self, radius):
        self.radii.append(radius)

    """Latent sizes"""
    def set_latent_sizes(self, sizes):
        self.latent_sizes = sizes

    def add_latent_size(self, size):
        self.latent_sizes.append(size)

    """Leaky relu"""
    def activate_leaky_relu(self):
        self.leaky = True

    def deactivate_leaky_relu(self):
        self.leaky = False

    """Complete encoding"""
    def make_complete_encoding(self):
        self.complete_encoding = True

    def make_incomplete_encoding(self):
        self.complete_encoding = False

    """Mean pooling"""
    def set_mean_pooling(self):
        self.mean = True

    def set_max_pooling(self):
        self.mean = False

    """Grid decoder"""
    def set_grid_decoder(self):
        self.grid = True

    def set_normal_decoder(self):
        self.grid = False

    """Split networks"""
    def enable_split_networks(self):
        self.split = True

    def disable_split_networks(self):
        self.split = False

    """Neighborhood encoder"""
    def get_neighborhood_encoder(self):
        latent_size = self.latent_sizes[0]
        return NeighborhoodEncoder(
            nbs_features=[32, 64, 64],
            nbs_features_global=[64, 32, latent_size],
            mean = self.mean,
            leaky=self.leaky
        )

    """Neighborhood decoder"""
    def get_neighborhood_decoder(self, layer):
        latent_size = self.latent_sizes[0]
        nb_neighbors = self.nbs_neighbors[layer]
        if self.grid:
            return GridDeformationDecoder(
                input_size=latent_size,
                nbs_features_global=[32, 64, 64],
                nbs_features=[64, 32, 3],
                nb_neighbors=nb_neighbors
            )
        else:
            NeighborhoodDecoder(
                input_size=latent_size,
                nbs_features_global=[32, 64, 64],
                nbs_features=[64, 32, 3],
                nb_neighbors=nb_neighbors
            )

    """Generate network"""
    def generate_network(self):
        encoders = []
        decoders = []
        encoders.append(self.get_neighborhood_encoder())
        decoders.append(self.get_neighborhood_decoder(0))
        if not self.split:
            encoder_class = MiddleLayerEncoder
        else:
            encoder_class = MiddleLayerEncoderSplit
        if self.nb_layers >= 2:
            encoders.append(
                encoder_class(
                    neighborhood_enc=self.get_neighborhood_encoder(),
                    input_size=self.latent_sizes[0],
                    nbs_features=[64, 128, 128],
                    nbs_features_global=[128, 64, self.latent_sizes[1]],
                    mean=self.mean,
                    leaky=self.leaky
                )
            )
        if self.nb_layers >= 3:
            encoders.append(
                encoder_class(
                    neighborhood_enc=self.get_neighborhood_encoder(),
                    input_size=self.latent_sizes[1],
                    nbs_features=[128, 256, 256],
                    nbs_features_global=[256, 128, self.latent_sizes[2]],
                    mean=self.mean,
                    leaky=self.leaky
                )
            )
        if not self.split:
            if self.nb_layers >= 2:
                decoders.append(
                    MiddleLayerDecoder(
                        neighborhood_dec=self.get_neighborhood_decoder(1),
                        input_size=self.latent_sizes[1],
                        nbs_features_global=[64, 128, self.latent_sizes[0]],
                        nbs_features=[64, 128, self.latent_sizes[0]],
                        leaky=self.leaky
                    )
                )
            if self.nb_layers >= 3:
                decoders.append(
                    MiddleLayerDecoder(
                        neighborhood_dec=self.get_neighborhood_decoder(2),
                        input_size=self.latent_sizes[2],
                        nbs_features_global=[128, 256, self.latent_sizes[0]],
                        nbs_features=[128, 256, self.latent_sizes[1]],
                        leaky=self.leaky
                    )
                )
        else:
            if self.nb_layers >= 2:
                decoders.append(
                    MiddleLayerDecoderSplit(
                        neighborhood_dec=self.get_neighborhood_decoder(1),
                        input_size=self.latent_sizes[1],
                        neigh_dec_size=self.latent_sizes[0],
                        nbs_features=[64, 128, self.latent_sizes[0]],
                        leaky=self.leaky
                    )
                )
            if self.nb_layers >= 2:
                decoders.append(
                    MiddleLayerDecoderSplit(
                        neighborhood_dec=self.get_neighborhood_decoder(2),
                        input_size=self.latent_sizes[2],
                        neigh_dec_size=self.latent_sizes[0],
                        nbs_features=[128, 256, self.latent_sizes[0] + self.latent_sizes[1]],
                        leaky=self.leaky
                    )
                )
        decoders.reverse()

        return PointCloudAE(
            nb_layers=self.nb_layers,
            nbs_neighbors=self.nbs_neighbors,
            radii=self.radii,
            encoders=encoders,
            decoders=decoders,
            complete_encoding=self.complete_encoding
        )



