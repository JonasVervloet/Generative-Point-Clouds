import torch
from torch import max
from torch.nn import Module, Linear
from torch_geometric.nn import fps
from torch_geometric.nn import knn
from torch_geometric.utils import scatter_


class SimpleFPSPoolLayer(Module):
    def __init__(self, nb_vertices):
        super(SimpleFPSPoolLayer, self).__init__()
        self.nb_vertices = nb_vertices

    def forward(self, data):
        result = data.pos
        i = 0

        while result.size(0) != self.nb_vertices and i < 100:
            ratio = self.nb_vertices / result.size(0)
            print(ratio)
            indices = fps(result, ratio=ratio)
            result = result[indices]
            i += 1

        assert(result.size(0) == self.nb_vertices)

        return result

