import torch
from torch import max, stack, cat
from torch.nn import Module, Linear
from torch_geometric.nn import fps
from torch_geometric.nn import knn
import random
import numpy as np


class RandomPoolLayer(Module):
    def __init__(self, nb_vertices):
        super(RandomPoolLayer, self).__init__()
        self.nb_vertices = nb_vertices

    def forward(self, x, batch=None):
        if batch is None:
            return self.select_random_features(x), batch
        else:
            curr_f = []
            curr_batch = []
            for i in range(max(batch) + 1):
                features = x[batch == i]
                pooled = self.select_random_features(features)
                curr_f.append(pooled)
                curr_batch.append(
                    torch.ones(pooled.size(0), dtype=torch.int64) * i
                )
            x = cat(curr_f)
            batch = cat(curr_batch)
            return x, batch

    def select_random_features(self, x):
        rand_inds = random.sample(
            range(x.size(0)), self.nb_vertices
        )
        return x[rand_inds]


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


class KnnUnpoolLayer(Module):
    def __init__(self, nb_outputs):
        super(KnnUnpoolLayer, self).__init__()
        self.outputs = nb_outputs

    def forward(self, x, batch=None):
        if batch is not None:
            curr_x = []
            curr_batch = []
            for i in range(max(batch) + 1):
                temp_batch = batch[batch == i]
                features = x[batch == i]
                rand_inds = np.random.randint(
                    features.size(0),
                    size=(self.outputs - features.size(0))
                )
                knn_neighbors = knn(features,
                                    features[rand_inds], 3)
                clusters = knn_neighbors[0]
                indices = knn_neighbors[1]
                result = []
                for j in range(max(clusters) + 1):
                    curr_inds = indices[clusters == j]
                    temp_feat = features[curr_inds]
                    result.append(temp_feat.mean(0))

                result = stack(result)
                new_batch = torch.ones(result.size(0), dtype=torch.int64) * i
                features = cat([features, result])
                temp_batch = cat([temp_batch, new_batch])
                curr_x.append(features)
                curr_batch.append(temp_batch)

            x = cat(curr_x)
            batch = cat(curr_batch)
            return x, batch

        else:
            rand_inds = np.random.randint(x.size(0), size=(self.outputs - x.size(0)))
            knn_neighbors = knn(x, x[rand_inds], 3)
            clusters = knn_neighbors[0]
            indices = knn_neighbors[1]
            result = []
            for i in range(max(clusters) + 1):
                curr_inds = indices[clusters == i]
                features = x[curr_inds]
                result.append(features.mean(0))

            result = stack(result)
            x = cat([x, result])

            return x, batch


class RandomUnpoolLayer(Module):
    def __init__(self, nb_outputs):
        super(RandomUnpoolLayer, self).__init__()
        self.outputs = nb_outputs

    def forward(self, x, batch=None):
        if batch is not None:
            curr_x = []
            curr_batch = []
            for i in range(max(batch) + 1):
                features = x[batch == i]
                new_features = self.rand_addition(features)
                new_batch = torch.ones(new_features.size(0), dtype=torch.int64) * i
                curr_x.append(new_features)
                curr_batch.append(new_batch)
            x = cat(curr_x)
            batch = cat(curr_batch)
            return x, batch

        else:
            return self.rand_addition(x), batch

    def rand_addition(self, x):
        rand_inds = np.random.randint(
            x.size(0),
            size=(self.outputs - x.size(0))
        )
        new_x = x[rand_inds]
        x = cat((x, new_x))
        return x
