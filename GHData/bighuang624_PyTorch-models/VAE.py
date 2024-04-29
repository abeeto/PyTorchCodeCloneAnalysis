# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")



class Encoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.sigmoid(self.linear2(x))


class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self._enc_mu = nn.Linear(100, latent_dim)
        self._enc_log_sigma = nn.Linear(100, latent_dim)
        self.z_mean = None
        self.z_sigma = None
        
    def _sample_latent(self, h_enc):
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        
        self.z_mean = mu
        self.z_sigma = sigma
        
        return mu + sigma * torch.tensor(std_z, requires_grad=False)    # Reparameterization
        
    def forward(self, x):
        h_enc = self.encoder(x)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_sigma):
    mean_sq = z_mean * z_mean
    sigma_sq = z_sigma * z_sigma
    return 0.5 * (mean_sq + sigma_sq - torch.log(sigma_sq) - 1)


if __name__ == '__main__':
    
    input_dim = 28 * 28
    latent_dim = 8
    batch_size = 32
    
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print('Number of samples:', len(mnist))
    
    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(latent_dim, 100, input_dim)
    vae = VAE(encoder, decoder, latent_dim)
    
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    l = None
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    
    for epoch in range(100):
        ll_sum = 0.0
        kl_sum = 0.0
        for i, data in enumerate(dataloader):
            inputs, classes = data
            inputs, classes = torch.tensor(inputs.resize_(batch_size, input_dim)), torch.tensor(classes)
            optimizer.zero_grad()
            outputs = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            kl = criterion(outputs, inputs)
            ll_sum += torch.mean(ll)
            kl_sum += torch.mean(kl)
            
        loss = ll_sum + kl_sum
        loss.backward()
        optimizer.step()
        l = loss.data[0]
            
        print('epoch: {}, loss:  {}'.format(epoch, l))
        
        #plt.imshow(inputs.data[0].numpy().reshape(28, 28), cmap='gray')
        #plt.show(block=True)
        plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
        plt.show(block=True)
    
    
    
    
    
    
    