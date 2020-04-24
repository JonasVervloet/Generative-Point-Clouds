import torch
import torch.nn as nn


class PointNetVAE(nn.Module):
    def __init__(self):
        super(PointNetVAE, self).__init__()
        self.encoder = None
        self.decoder = None
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


