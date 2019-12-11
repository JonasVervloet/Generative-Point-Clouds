import torch
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from PoolLayer import RandomPoolLayer
from torch_geometric.data import Data


class ShapeNetFunctionality:

    @staticmethod
    def histogram(dataset, name, nb_bins=None):
        nb_nodes = ShapeNetFunctionality.get_number_of_nodes(dataset)
        if nb_bins is not None:
            plt.hist(nb_nodes, bins=nb_bins)
        else:
            plt.hist(nb_nodes)
        plt.title("Nb Points: " + name)
        plt.show()

    @staticmethod
    def get_total_nb_nodes(dataset):
        nb_nodes = ShapeNetFunctionality.get_number_of_nodes(dataset)
        return sum(nb_nodes)

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

    @staticmethod
    def simple_resample_to_minimum(dataset):
        minimum = min(ShapeNetFunctionality.
                      get_number_of_nodes(dataset))
        pool = RandomPoolLayer(minimum)
        result = []
        i = 0
        for data in dataset:
            vertices, _ = pool(data.pos)
            result.append(
                Data(pos=vertices)
            )
        return result

    @staticmethod
    def random_split_ratio(dataset, ratio):
        nb_set1 = round(len(dataset) * ratio)
        set1 = sample(dataset, nb_set1)
        set2 = [data for data in dataset if data not in set1]
        return set1, set2

    @staticmethod
    def reshape_batch(batch, nb_vertices):
        return batch.view(-1, nb_vertices, batch.size(1))

    @staticmethod
    def dist_between_points(point1, point2):
        return sum((point1-point2)**2)



