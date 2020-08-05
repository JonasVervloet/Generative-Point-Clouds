import torch
from torch import max, stack, cat
from torch.nn import Module, Linear
from torch_geometric.nn import fps
from torch_geometric.nn import knn
import random
import numpy as np
from torch_geometric.utils import scatter_


class PoolLayer(Module):
    def __init__(self, nb_outputs, pool_fn):
        super(PoolLayer, self).__init__()
        self.outputs = nb_outputs
        self.function = pool_fn

    def forward(self, x, batch=None):
        if batch is not None:
            curr_x = []
            curr_batch = []
            for i in range(max(batch) + 1):
                features = x[batch == i]
                new_features = self.function(features)
                new_batch = torch.ones(new_features.size(0), dtype=torch.int64) * i
                curr_x.append(new_features)
                curr_batch.append(new_batch)
            x = cat(curr_x)
            batch = cat(curr_batch)
            return x, batch

        else:
            return self.function(x), batch


class RandomPoolLayer(PoolLayer):
    def __init__(self, nb_vertices):
        super(RandomPoolLayer, self).__init__(
            nb_vertices, self.select_random_features
        )

    def select_random_features(self, x):
        # rand_inds = random.sample(
        #     range(x.size(0)), self.outputs
        # )
        perm = torch.randperm(x.size(0))
        idx = perm[:self.outputs]
        return x[idx]


class TopKPool(PoolLayer):
    def __init__(self, nb_outputs):
        super(TopKPool, self).__init__(nb_outputs, self.max_pooling)

    def max_pooling(self, x):
        dists = torch.sum(x**2, dim=1)
        values, inds = torch.topk(dists, self.outputs, largest=True)
        return x[inds]


class RandomKNNPool(PoolLayer):
    def __init__(self, nb_outputs):
        super(RandomKNNPool, self).__init__(nb_outputs, self.random_knn_pooling)

    def random_knn_pooling(self, x):
        rand_inds = random.sample(
            range(x.size(0)), self.outputs
        )
        neighbors = knn(x, x[rand_inds], 3)
        cluster, indices = neighbors
        cluster_feats = x[indices]
        new_features = scatter_("mean", cluster_feats, cluster)
        return new_features


class FPSPoolLayer(Module):
    def __init__(self, nb_vertices):
        super(FPSPoolLayer, self).__init__()
        self.nb_vertices = nb_vertices

    def forward(self, x, batch=None):
        ratio = 0
        if batch is not None:
            ratio = self.nb_vertices / batch[batch == 0].size(0)
        else:
            ratio = self.nb_vertices / x.size(0)
        if ratio == 1:
            return x
        else:
            indices = fps(x, ratio=ratio, batch=batch)
            x = x[indices]
            if batch is not None:
                batch = batch[indices]
            return x, batch


class KnnUnpoolLayer(PoolLayer):
    def __init__(self, nb_outputs):
        super(KnnUnpoolLayer, self).__init__(
            nb_outputs, self.knn_unpool
        )

    def knn_unpool(self, x):
        feats = x
        while self.outputs >= 2 * feats.size(0):
            feats = self.knn_aggreate(feats, feats)
        if feats.size(0) == self.outputs:
            return feats
        else:
            nb = self.outputs - feats.size(0)
            rand_inds = random.sample(
                range(feats.size(0)), nb
            )
            return self.knn_aggreate(feats, feats[rand_inds])

    def knn_aggreate(self, features, goals):
        cluster, indices = knn(features, goals, 3)
        cluster_feats = features[indices]
        new_features = scatter_("mean", cluster_feats, cluster)
        return torch.cat([features, new_features])


class KnnUnpoolLayer2(PoolLayer):
    def __init__(self, nb_outputs):
        super(KnnUnpoolLayer2, self).__init__(
            nb_outputs, self.knn_unpool
        )

    def knn_unpool(self, x):
        feats = x
        while not feats.size(0) == self.outputs:
            ind = random.sample(
                range(feats.size(0)), 1
            )
            cluster, indices = knn(feats, feats[ind], 3)
            cluster_feats = feats[indices]
            new_features = scatter_("mean", cluster_feats, cluster)
            feats = torch.cat([feats, new_features])
        return feats


class RandomUnpoolLayer(PoolLayer):
    def __init__(self, nb_outputs):
        super(RandomUnpoolLayer, self).__init__(
            nb_outputs, self.rand_addition
        )

    def rand_addition(self, x):
        # rand_inds = np.random.randint(
        #     x.size(0),
        #     size=(self.outputs - x.size(0))
        # )
        rand_inds = torch.randint(
            0, x.size(0), (self.outputs - x.size(0),)
        )
        new_x = x[rand_inds]
        x = cat((x, new_x))
        return x


