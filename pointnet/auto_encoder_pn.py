import torch
import torch.nn as nn
from PointNetPP import PointNetPP
from Decoder import Decoder


class PointNetAE(nn.Module):
    def __init__(self):
        super(PointNetAE, self).__init__()
        self.encoder = PointNetPP(0.1, 0.1, 10)
        self.decoder = Decoder(5, 10)

    def forward(self, points, batch):

        # points = (nb_batch * nb_points) x 3
        # batch = (nb_batch * nb_points)
        latent = self.encoder(points, batch)
        print(latent.size())
        # latent = nb_batch x 1024
        n_points = self.decoder(latent)
        # n_points = (nb_batch * 20 * 4 * 4) x 3

        return n_points


class PointNetVAE(nn.Module):
    def __init__(self):
        super(PointNetVAE, self).__init__()
        self.encoder = PointNetPP(0.1, 0.1, 10)
        self.decoder = Decoder(20, 10)
        self.mu = nn.Linear(256, 256)
        self.var = nn.Linear(256, 256)

    def forward(self, points, batch):
        latent = self.encoder(points, batch)

        z_mu = self.mu(latent)
        z_var = self.var(latent)
        std = torch.exp(z_var/2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add(z_mu)

        n_points = self.decoder(x_sample)

        return n_points, z_mu, z_var

    def evaluate(self, points, batch):
        latent = self.encoder(points, batch)

        z_mu = self.mu(latent)
        z_var = self.var(latent)

        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add(z_mu)

        return x_sample

    def decode(self, latent):
        return self.decoder(latent)


