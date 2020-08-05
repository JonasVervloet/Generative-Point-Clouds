import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')
        self.mlp1 = Linear(in_channels, out_channels)
        self.mlp2 = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        size = x.size(0)
        return self.propagate(edge_index, size=(size, size), x=x)

    def message(self, x_i, x_j):
        val1 = self.mlp1(x_i)
        val2 = self.mlp2(x_j - x_i)

        return val1 + val2

    def update(self, aggr_out):
        return aggr_out


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)
