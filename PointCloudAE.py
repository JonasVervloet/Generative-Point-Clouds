import torch
from torch.nn import Module, Linear
from DynamicEdgeConv import DynamicEdgeConv
from PoolLayer import FPSPoolLayer, RandomPoolLayer, KnnUnpoolLayer, RandomUnpoolLayer


class PointCloudAE(Module):
    def __init__(self):
        super(PointCloudAE, self).__init__()
        self.conv1 = DynamicEdgeConv(3, 9, 3)
        self.pool1 = FPSPoolLayer(500)
        self.conv2 = DynamicEdgeConv(9, 12, 3)
        self.pool2 = FPSPoolLayer(50)
        self.conv3 = DynamicEdgeConv(12, 12, 3)
        self.lin1 = Linear(50 * 12, 30)

        self.lin2 = Linear(30, 50 * 12)
        self.conv4 = DynamicEdgeConv(12, 12, 3)
        self.unpool1 = KnnUnpoolLayer(500)
        self.conv5 = DynamicEdgeConv(12, 9, 3)
        self.unpool2 = KnnUnpoolLayer(2300)
        self.conv6 = DynamicEdgeConv(9, 3, 3)

    def forward(self, x, batch=None):
        x, batch = self.encode(x, batch)
        x = self.decode(x, batch)
        return x

    def encode(self, x, batch=None):
        x = self.conv1(x, batch)
        x, batch = self.pool1(x, batch)
        x = self.conv2(x, batch)
        x, batch = self.pool2(x, batch)
        x = self.conv3(x, batch)
        if batch is None:
            x = x.reshape(-1)
        else:
            nb = max(batch) + 1
            x = x.reshape(nb, -1)
        x = self.lin1(x)
        return x, batch

    def decode(self, x, batch=None):
        x = self.lin2(x)
        if batch is None:
            x = x.reshape(50, 12)
        else:
            nb = max(batch) + 1
            x = x.reshape(nb * 50, -1)
        x = self.conv4(x, batch)
        x, batch = self.unpool1(x, batch)
        x = self.conv5(x, batch)
        x, batch = self.unpool2(x, batch)
        x = self.conv6(x, batch)
        return x

class PointCoudAERandom(Module):
    def __init__(self):
        super(PointCoudAERandom, self).__init__()
        self.conv1 = DynamicEdgeConv(3, 9, 3)
        self.pool1 = RandomPoolLayer(500)
        self.conv2 = DynamicEdgeConv(9, 12, 3)
        self.pool2 = RandomPoolLayer(50)
        self.conv3 = DynamicEdgeConv(12, 12, 3)
        self.lin1 = Linear(50 * 12, 30)

        self.lin2 = Linear(30, 50 * 12)
        self.conv4 = DynamicEdgeConv(12, 12, 3)
        self.unpool1 = RandomUnpoolLayer(500)
        self.conv5 = DynamicEdgeConv(12, 9, 3)
        self.unpool2 = RandomUnpoolLayer(2300)
        self.conv6 = DynamicEdgeConv(9, 3, 3)

    def forward(self, x, batch=None):
        x, batch = self.encode(x, batch)
        x = self.decode(x, batch)
        return x

    def encode(self, x, batch=None):
        x = self.conv1(x, batch)
        x, batch = self.pool1(x, batch)
        x = self.conv2(x, batch)
        x, batch = self.pool2(x, batch)
        x = self.conv3(x, batch)
        if batch is None:
            x = x.reshape(-1)
        else:
            nb = max(batch) + 1
            x = x.reshape(nb, -1)
        x = self.lin1(x)
        return x, batch

    def decode(self, x, batch=None):
        x = self.lin2(x)
        if batch is None:
            x = x.reshape(50, 12)
        else:
            nb = max(batch) + 1
            x = x.reshape(nb * 50, -1)
        x = self.conv4(x, batch)
        x, batch = self.unpool1(x, batch)
        x = self.conv5(x, batch)
        x, batch = self.unpool2(x, batch)
        x = self.conv6(x, batch)
        return x




