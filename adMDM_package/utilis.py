import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        # x_dim: dimension of the input feature
        # z_dim: dimension of the latent
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(self.x_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
        )

    def forward(self, x):
        latent = self.net(x)

        return latent  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, self.x_dim)
        )

    def forward(self, z):
        x = self.net(z)
        return x


class Classifier(nn.Module):
    def __init__(self, z_dim):
        super(Classifier, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1)
            # nn.Flatten()
        )

    def forward(self, z):
        critic = self.net(z)
        return critic
