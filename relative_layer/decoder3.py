import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import math


class RelativeDecoder2(nn.Module):
    def __init__(self, nb_neighbours, nb_feats_in, nb_feats_middle, nb_feats_middle2):
        super(RelativeDecoder2, self).__init__()

        self.neighs = nb_neighbours

        self.fc1 = nn.Linear(nb_feats_in + 2, nb_feats_middle)
        self.fc2 = nn.Linear(nb_feats_middle, nb_feats_middle2)
        self.fc3 = nn.Linear(nb_feats_middle2, 3)

        self.fc4 = nn.Linear(nb_feats_in + 3, nb_feats_middle)
        self.fc5 = nn.Linear(nb_feats_middle, nb_feats_middle2)
        self.fc6 = nn.Linear(nb_feats_middle2, 3)

    def forward(self, feats):
        # feats = nb_cluster x nb_feats_in

        # repeated = (nb_cluster * nb_neighbours) x nb_feats_in
        repeated = torch.repeat_interleave(feats, self.neighs, dim=0)

        # grids = (nb_cluster * nb_neighbours) x 2
        grids = self.create_grid().repeat((feats.size(0), 1))

        # concat = (nb_cluster * nb_neighbours) x (nb_feats_in + 2)
        concat = torch.cat([repeated, grids], dim=1)

        # fc1 = (nb_cluster * nb_neighbours) x nb_feats_middle
        fc1 = torch.relu(self.fc1(concat))

        # fc2 = (nb_cluster * nb_neighbours) x nb_feats_middle2
        fc2 = torch.relu(self.fc2(fc1))

        # fc3 = (nb_cluster * nb_neighbours) x 3
        fc3 = torch.tanh(self.fc3(fc2))

        # concat2 = (nb_cluster * nb_neighbours) x (nb_feats_in + 2)
        concat2 = torch.cat([repeated, fc3], dim=1)

        # fc4 = (nb_cluster * nb_neighbours) x nb_feats_middle
        fc4 = torch.relu(self.fc4(concat2))

        # fc5 = (nb_cluster * nb_neighbours) x nb_feats_middle2
        fc5 = torch.relu(self.fc5(fc4))

        # out = (nb_cluster * nb_neighbours) x 3
        return torch.tanh(self.fc6(fc5))

    def eval(self, feats):
        # feats = nb_cluster x nb_feats_in

        # repeated = (nb_cluster * nb_neighbours) x nb_feats_in
        repeated = torch.repeat_interleave(feats, self.neighs, dim=0)

        # grids = (nb_cluster * nb_neighbours) x 2
        grids = self.create_grid().repeat((feats.size(0), 1))

        # concat = (nb_cluster * nb_neighbours) x (nb_feats_in + 2)
        concat = torch.cat([repeated, grids], dim=1)

        # fc1 = (nb_cluster * nb_neighbours) x nb_feats_middle
        fc1 = torch.relu(self.fc1(concat))

        # fc2 = (nb_cluster * nb_neighbours) x nb_feats_middle2
        fc2 = torch.relu(self.fc2(fc1))

        # fc3 = (nb_cluster * nb_neighbours) x 3
        fc3 = torch.tanh(self.fc3(fc2))

        # concat2 = (nb_cluster * nb_neighbours) x (nb_feats_in + 2)
        concat2 = torch.cat([repeated, fc3], dim=1)

        # fc4 = (nb_cluster * nb_neighbours) x nb_feats_middle
        fc4 = torch.relu(self.fc4(concat2))

        # fc5 = (nb_cluster * nb_neighbours) x nb_feats_middle2
        fc5 = torch.relu(self.fc5(fc4))

        # out1 = (nb_cluster * nb_neighbours) x 2
        # out2 = (nb_cluster * nb_neighbours) x 3
        # out3 = (nb_cluster * nb_neighbours) x 3
        return grids, fc3, torch.tanh(self.fc6(fc5))

    def create_grid(self):
        nb = int(math.sqrt(self.neighs))
        dist = 1/nb
        grid = []
        for i in range(nb):
            for j in range(nb):
                grid.append(
                    dist * torch.tensor(
                        [i + 0.5, j + 0.5]
                    )
                )
        return torch.stack(grid)

