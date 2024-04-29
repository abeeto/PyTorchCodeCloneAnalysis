




import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.utils



latent_dims = 2
variational_beta = 1.5


class Encoder(nn.Module):
    def __init__(self, capacity):
        super(Encoder, self).__init__()
        self.capacity = c = capacity
        # input: [B, C, H, W] -- H, W = 112
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)     # out = c x 1 x 56 x 56
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1)   # out = c x 2 x 28 x 28
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1) # out = c x 4 x 14 x 14
        self.conv4 = nn.Conv2d(in_channels=c*4, out_channels=c*8, kernel_size=4, stride=2, padding=1) # out = c x 8 x 7 x 7
        ##self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # dimension/2
        self.conv5 = nn.Conv2d(in_channels=c*8, out_channels=c*1, kernel_size=1, stride=1, padding=0) # out = c x 1 x 7 x 7

        self.fc_mu = nn.Linear(in_features=c*1*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*1*7*7, out_features=latent_dims)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.30)

        self.gradients = None


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        
        if (x.requires_grad == True) and (self.training == False):
            x = self.conv5(x)
            # hook the gradient
            g = x.register_hook(self.activations_hook)
            x = F.relu(x) 

        else:
            x = F.relu(self.conv5(x))

        # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = x.view(x.size(0), -1) 
        #x = F.relu(self.fc2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, capacity):
        super(Decoder, self).__init__()
        self.capacity = c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*1*7*7)

        self.conv5 = nn.ConvTranspose2d(in_channels=c*1, out_channels=c*8, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*8, out_channels=c*4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.30)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*1, 7, 7) ## unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x



class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity, training_mode):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(capacity)
        self.decoder = Decoder(capacity)

        self.training = training_mode

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar) # sample function is at below
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            #print("Model: Training mode...")
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            print("Model: Testing mode...")
            return mu



def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error




def vae_loss_true_size(recon_x, x, true_size, beta, mu, logvar):
    # recon_x: the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy
    # averaging or not averaging the binary cross-entropy over all pixels here 
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelyhood,
    # but averaging makes the weight of the other loss term independent of the image resolution
    recon_loss = F.binary_cross_entropy(x.view(-1, true_size), recon_x.view(-1, true_size), reduction="sum")

    # KL-divergence between the prior distribution over latent vectors
    # (the one we going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #print("  recon_loss: {}, kd-d: {}".format(recon_loss, kldivergence) )

    #return recon_loss + beta*kldivergence
    return recon_loss.mean(), beta*kldivergence.mean()




















