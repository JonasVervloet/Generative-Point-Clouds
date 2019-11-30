import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt


class ShapeNetFunctionality:

    @staticmethod
    def histogram(dataset, name):
        nb_nodes = ShapeNetFunctionality.get_number_of_nodes(dataset)
        plt.hist(nb_nodes)
        plt.title("Nb Points: " + name)
        plt.show()

    @staticmethod
    def get_number_of_nodes(dataset):
        nb_nodes = np.zeros(len(dataset), dtype=int)
        for i in range(len(dataset)):
            nb_nodes[i] = dataset[i].num_nodes

        return nb_nodes

    @staticmethod
    def filter_data(dataset, min_nb):
        filtered = []
        for data in dataset:
            if data.num_nodes >= min_nb:
                filtered.extend([data])
        return filtered

