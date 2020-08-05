import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class RelativeDecoder(nn.Module):
    def __init__(self, feats_in, feats1, feats2, neighs):
        super(RelativeDecoder, self).__init__()

        self.fc1 = nn.Linear(feats_in, feats1)
        self.fc2 = nn.Linear(feats1, feats2)
        self.fc3 = nn.Linear(feats2, feats2 * neighs)
        self.fc4 = nn.Linear(feats2, 3)

        self.neighs = neighs

    def forward(self, feats):
        # feats = nb_cluster x (nb_feats_in + 2)
        nb_cluster = feats.size(0)
        alfa_vec = feats[:, 0]
        beta_vec = feats[:, 1]
        gamma_vec = feats[:, 2]
        rest = feats[:, 3:]

        # fc1 = nb_cluster x nb_feats1
        fc1 = F.relu(self.fc1(rest))

        # fc2 = nb_cluster x nb_feats2
        fc2 = F.relu(self.fc2(fc1))

        # fc3 = nb_cluster x (nb_feats2 * nb_neighs)
        fc3 = F.relu(self.fc3(fc2))

        # resized = nb_cluster x nb_neighs x nb_feats2
        resized = fc3.view(nb_cluster, self.neighs, -1)

        # fc4 = nb_cluster x nb_neighs x 3
        fc4 = torch.tanh(self.fc4(resized))

        # rot_matrices = nb_cluster x 3 x 3
        rot_matrices = self.generate_rotation_matrices(
            alfa_vec=alfa_vec, beta_vec=beta_vec, gamma_vec=gamma_vec
        )

        # rotated = nb_cluster x nb_neighs x 3
        rotated = torch.bmm(fc4, rot_matrices)

        # out = (nb_cluster * nb_neighs) x 3
        return rotated.view(-1, 3)

    def eval(self, feats):
        # feats = nb_cluster x (nb_feats_in + 2)
        nb_cluster = feats.size(0)
        alfa_vec = feats[:, 0]
        beta_vec = feats[:, 1]
        gamma_vec = feats[:, 2]
        rest = feats[:, 3:]

        # fc1 = nb_cluster x nb_feats1
        fc1 = F.relu(self.fc1(rest))

        # fc2 = nb_cluster x nb_feats2
        fc2 = F.relu(self.fc2(fc1))

        # fc3 = nb_cluster x (nb_feats2 * nb_neighs)
        fc3 = F.relu(self.fc3(fc2))

        # resized = nb_cluster x nb_neighs x nb_feats2
        resized = fc3.view(nb_cluster, self.neighs, -1)

        # fc4 = nb_cluster x nb_neighs x 3
        fc4 = torch.tanh(self.fc4(resized))

        # rot_matrices = nb_cluster x 3 x 3
        rot_matrices = self.generate_rotation_matrices(
            alfa_vec=alfa_vec, beta_vec=beta_vec, gamma_vec=gamma_vec
        )

        # rotated = nb_cluster x nb_neighs x 3
        rotated = torch.bmm(fc4, rot_matrices)

        # out1 = (nb_cluster * nb_neighs) x 3
        # out2 = (nb_cluster * nb_neighs) x 3
        return fc4.view(-1, 3), rotated.view(-1, 3)

    @staticmethod
    def generate_rotation_matrices(alfa_vec, beta_vec, gamma_vec):
        assert(alfa_vec.size(0) == beta_vec.size(0))
        assert(alfa_vec.size(0) == gamma_vec.size(0))
        matrices = []

        for i in range(alfa_vec.size(0)):
            alfa = alfa_vec[i] * math.pi * 2
            beta = beta_vec[i] * math.pi * 2
            gamma = gamma_vec[i] * math.pi * 2

            x_rotation = torch.tensor(
                [[1, 0, 0],
                 [0, torch.cos(alfa), -torch.sin(alfa)],
                 [0, torch.sin(alfa), torch.cos(alfa)]]
            )
            y_rotation = torch.tensor(
                [[torch.cos(beta), 0, torch.sin(beta)],
                 [0, 1, 0],
                 [-torch.sin(beta), 0, torch.cos(beta)]]
            )
            z_rotation = torch.tensor(
                [[torch.cos(gamma), -torch.sin(gamma), 0],
                 [torch.sin(gamma), torch.cos(gamma), 0],
                 [0, 0, 1]]
            )

            matrix = torch.mm(z_rotation, torch.mm(y_rotation, x_rotation))
            matrices.append(matrix)

        return torch.stack(matrices)




