from .utilis import *
from torch.distributions import Normal, Independent
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns


def inner_prod(x, y):
    sim = torch.matmul(x, torch.transpose(y, 0, 1))
    return torch.trace(sim)


class adMDM(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, device):
        super(adMDM, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.device = device

        # define the network structure
        self.generator_x = Encoder(self.x_dim, self.z_dim)
        self.generator_y = Encoder(self.y_dim, self.z_dim)

        self.decoder_x = Decoder(self.x_dim, self.z_dim)
        self.decoder_y = Decoder(self.y_dim, self.z_dim)

        self.discriminator = Classifier(self.z_dim)

    def train_discriminator(self, x, y, opt_d, lambda_gp=10):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass x through discriminator
        latent_x = self.generator_x(x)
        preds_x = self.discriminator(latent_x)
        loss_x = - torch.mean(preds_x)

        # Pass y through discriminator
        latent_y = self.generator_y(y)
        preds_y = self.discriminator(latent_y)
        loss_y = torch.mean(preds_y)

        # Gradient Penalty
        # from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        # from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
        alpha = torch.rand(latent_y.shape[0], 1)
        alpha = alpha.expand(latent_y.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * latent_y + ((1 - alpha) * latent_x)
        interpolates = autograd.Variable(interpolates, requires_grad=True).to(self.device)
        disc_interpolates = self.discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = loss_x + loss_y + lambda_gp * gradient_penalty

        loss.backward()
        opt_d.step()

        return loss_x.item(), loss_y.item()

    def train_generator_x_with_simloss(self, x, y, opt_gx,
                                       sim_lambd=1):
        # Clear generator gradients
        opt_gx.zero_grad()

        # Pass x through discriminator
        latent_x = self.generator_x(x)
        preds_x = self.discriminator(latent_x)
        loss_x = torch.mean(preds_x)

        latent_y = self.generator_y(y)

        simloss = inner_prod(latent_x, latent_y)
        loss = loss_x - sim_lambd * simloss

        # Update discriminator weights
        # loss_x.backward()
        loss.backward()
        opt_gx.step()

        return loss.item()

    def train_generator_y_with_simloss(self, x, y, opt_gy,
                                       sim_lambd=1):
        # Clear generator gradients
        opt_gy.zero_grad()

        # Pass y through discriminator
        latent_y = self.generator_y(y)
        preds_y = self.discriminator(latent_y)
        loss_y = -torch.mean(preds_y)  # attention: take a negative

        latent_x = self.generator_x(x)

        simloss = inner_prod(latent_x, latent_y)
        loss = loss_y - sim_lambd * simloss

        # Update discriminator weights
        # loss_x.backward()
        loss.backward()
        opt_gy.step()

        return loss.item()

    def train_autoencoder_x(self, x, opt_ax):
        # Clear autoencoder gradients
        opt_ax.zero_grad()

        # Generators
        z_x = self.generator_x(x)
        # Decoder
        x_hat = self.decoder_x(z_x)
        # Reconstruction
        recon_loss = F.mse_loss(x_hat, x)
        recon_loss = recon_loss.mean()

        # Update autoencoder weights
        loss = recon_loss
        loss.backward()
        opt_ax.step()

        return loss.item()

    def train_autoencoder_y(self, y, opt_ay):
        # Clear autoencoder gradients
        opt_ay.zero_grad()

        # Generators
        z_y = self.generator_y(y)
        # Decoder
        y_hat = self.decoder_y(z_y)
        # Reconstruction
        recon_loss = F.mse_loss(y_hat, y)
        recon_loss = recon_loss.mean()

        # Update autoencoder weights
        loss = recon_loss
        loss.backward()
        opt_ay.step()

        return loss.item()

    def fit(self, train_dl, epochs=2, lr=1e-3, epochs_ae=200,
            print_interval=100, step_ae=2, step_g=5, step_d=2, weight_decay=0, epochs_inner=2001,
            epoch_ad=10):
        torch.cuda.empty_cache()

        # Losses & scores
        losses_gx = []
        losses_gy = []

        losses_ax = []
        losses_ay = []

        losses_d = []

        # Define optimizers
        # 1. autoencoders
        opt_ax_params = list(self.generator_x.parameters()) + list(self.decoder_x.parameters())
        opt_ax = torch.optim.Adam(opt_ax_params, lr=lr,
                                  weight_decay=weight_decay, betas=(0.5, 0.9), amsgrad=True)

        opt_ay_params = list(self.generator_y.parameters()) + list(self.decoder_y.parameters())
        opt_ay = torch.optim.Adam(opt_ay_params, lr=lr,
                                  weight_decay=weight_decay, betas=(0.5, 0.9), amsgrad=True)

        # 2. discriminator
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99),
                                 amsgrad=True)

        # 3. generator
        opt_gx = torch.optim.Adam(self.generator_x.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.5, 0.9),
                                  amsgrad=True)
        opt_gy = torch.optim.Adam(self.generator_y.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.5, 0.9),
                                  amsgrad=True)

        # Optimization of autoencoder
        print('Optimize autoencoder losses')
        for epoch in range(epochs_ae):
            for x, y in train_dl:
                # Train autoencoder
                loss_ax = self.train_autoencoder_x(x, opt_ax)
                loss_ay = self.train_autoencoder_y(y, opt_ay)
                losses_ax.append(loss_ax)
                losses_ay.append(loss_ay)
            if epoch % print_interval == 0:
                print(
                    "Epoch [{}/{}]\n\tloss_ax: {:.4f}\tloss_ay: {:.4f}".format(
                        epoch + 1, epochs_ae, loss_ax, loss_ay))

        # Optimization of all losses
        print('Optimize adversarial and autoencoder losses')

        # Assign initial values
        loss_d = 0
        loss_gx = 0
        loss_gy = 0

        for epoch in range(epochs):
            # inner_size = int(epochs_inner / (epoch + 1))
            inner_size = int(epochs_inner / (np.log(epoch + 1) + 1))
            # Fix y and optimize x
            for epoch_inner in range(inner_size):
                for x, y in train_dl:
                    # Train autoencoder
                    for step in range(step_ae):
                        loss_ax = self.train_autoencoder_x(x, opt_ax)
                    losses_ax.append(loss_ax)

                    for kk in range(epoch_ad):
                        # Train discriminator
                        for step in range(step_d):
                            loss_d, _ = self.train_discriminator(x, y, opt_d)
                        losses_d.append(loss_d)
                        # Train generator
                        for step in range(step_g):
                            loss_gx = self.train_generator_x_with_simloss(
                                x, y, opt_gx, sim_lambd=1)
                        losses_gx.append(loss_gx)
                # Log losses & scores (last batch)
                if epoch_inner % print_interval == 0:
                    print(
                        "x Epoch [{}/{}]\n\tloss_gx: {:.4f}\n\tloss_dx: {:.4f}\n\tloss_ay: {:.4f}".format(
                            epoch_inner + 1, epochs_inner, loss_gx, loss_d, loss_ax))

            # Fix x and optimize y
            for epoch_inner in range(inner_size):
                for x, y in train_dl:
                    # Train autoencoder
                    for step in range(step_ae):
                        loss_ay = self.train_autoencoder_y(y, opt_ay)
                    losses_ay.append(loss_ay)

                    for kk in range(epoch_ad):
                        # Train discriminator
                        for step in range(step_d):
                            loss_d, _ = self.train_discriminator(x, y, opt_d)
                        losses_d.append(loss_d)
                        # Train generator
                        for step in range(step_g):
                            loss_gy = self.train_generator_y_with_simloss(
                                x, y, opt_gy, sim_lambd=1)
                        losses_gy.append(loss_gy)
                if epoch_inner % print_interval == 0:
                    print(
                        "y Epoch [{}/{}]\n\tloss_gy: {:.4f}\n\tloss_dy: {:.4f}\n\tloss_ay: {:.4f}".format(
                            epoch_inner + 1, epochs_inner, loss_gy, loss_d, loss_ay))

        return losses_ax, losses_gx, losses_d

    def inference(self, x, y):
        latent_x = self.generator_x(x)
        latent_y = self.generator_y(y)

        return latent_x, latent_y
