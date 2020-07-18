import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from relative_layer.neighborhood_encoder import NeighborhoodEncoder
from relative_layer.neighborhood_decoder import NeighborhoodDecoder
from relative_layer.grid_deform_decoder import GridDeformationDecoder
from full_network.middlelayer_encoder import MiddleLayerEncoder, MiddleLayerEncoderSplit
from full_network.middlelayer_decoder import MiddleLayerDecoder, MiddleLayerDecoderSplit
from full_network.point_cloud_ae import PointCloudAE

from dataset.primitive_shapes import PrimitiveShapes

from loss_function import ChamferDistLoss, ChamferVAELoss, LayerChamferDistLoss, ChamferDistLossFullNetwork


class NetworkTextFileConverter:
    def __init__(self, map_path):
        self.path = map_path

        self.networks = [
            NeighborhoodEncoder,
            NeighborhoodDecoder,
            GridDeformationDecoder,
            MiddleLayerEncoder,
            MiddleLayerEncoderSplit,
            MiddleLayerDecoder,
            MiddleLayerDecoderSplit,
            PointCloudAE
        ]

        self.dataset_generators = [
            PrimitiveShapes
        ]

        self.loss_functions = [
            ChamferDistLoss,
            ChamferVAELoss,
            LayerChamferDistLoss,
            ChamferDistLossFullNetwork
        ]

        self.optimizers = [
            Adam
        ]

    def load_manager(self, manager):
        read_file = open(self.path + "info.txt", "r")
        lines = read_file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n", "")

        return manager.from_string(lines, self, manager)

    def load_network(self, network, epoch):
        network.load_state_dict(
            torch.load(self.path + "model_epoch{}.pt".format(epoch))
        )

    def load_losses(self, epoch, train):
        if train:
            category = "train"
        else:
            category = "val"
        return np.load(self.path + category + "loss_epoch{}.npy".format(epoch))

    def read_network_lines(self, string_lines):
        for network in self.networks:
            try:
                return network.from_string(string_lines, self)
            except AssertionError:
                continue

        raise ValueError(
            "This Network String Reader is not able to read the input lines"
        )

    def read_generator_lines(self, string_lines):
        for generator in self.dataset_generators:
            try:
                return generator.from_string(string_lines, self)
            except AssertionError:
                continue

        raise ValueError(
            "This network converter is not able to read the input dataset generator lines!"
        )

    def read_loss_function_lines(self, string_lines):
        for loss_function in self.loss_functions:
            try:
                return loss_function.from_string(string_lines)
            except AssertionError:
                continue

        raise ValueError(
            "This network converter is not able to read the input loss function lines!"
        )

    def save_network(self, network, epoch_nb):
        torch.save(
            network.state_dict(),
            self.path + "model_epoch{}.pt".format(epoch_nb)
        )

    def save_losses(self, losses, epoch_nb, train=True):
        if train:
            category = "train"
        else:
            category = "val"

        np.save(
            self.path + category + "loss_epoch{}.npy".format(epoch_nb),
            losses
        )

    def save_loss_plot(self, train_loss, val_losses, epoch_nb):
        assert(len(train_loss) == epoch_nb + 1)
        assert(len(val_losses) == epoch_nb + 1)

        plt.clf()
        x = range(epoch_nb + 1)
        plt.plot(x, train_loss, x, val_losses)
        plt.legend(['train loss', 'validation loss'])
        plt.title('Point AutoEncoder Network Losses')
        plt.yscale('log')
        plt.savefig(
            self.path + "loss_epoch{}.png".format(epoch_nb)
        )

    def save_info_file(self, network_manager):
        info_file = open(self.path + "info.txt", "w")
        info_file.write(network_manager.to_string())
        info_file.close()

    def get_optimizer(self, input_line):
        for optim in self.optimizers:
            if str(optim.__name__) == input_line:
                return optim

        raise ValueError(
            "This network converter is not able to read the input optimizer line!"
        )



