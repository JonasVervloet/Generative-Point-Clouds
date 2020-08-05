import torch
from torch.nn import Module
from torch_geometric.nn import fps


def sample_and_group(pos, feats, batch=None, ratio=0.5, region_size=0.5):
    print(pos.size())
    print(feats.size())
    print(ratio)
    samples = fps(pos, batch=batch, ratio=ratio)
    print(samples.size())
    print(samples)


class EncodeLayer(Module):
    def __init__(self, ratio):
        super(EncodeLayer, self).__init__()
        self.ratio = ratio

    def forward(self, pos, feats, batch=None):
        print(pos.size())
        print(feats.size())
        print(self.ratio)
        samples = fps(pos, batch=batch, ratio=self.ratio)
        print(samples.size())
        print(samples)


