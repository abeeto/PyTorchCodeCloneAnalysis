# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 00:38:32 2022

@author: HP_PC2
"""

import matplotlib.pyplot as plt
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Load and prepare data
train_transform = transforms.Compose([transforms.ToTensor(),])
test_transform = transforms.Compose([transforms.ToTensor(),])

train_dataset = torchvision.datasets.MNIST(root='/data', train=True,
                                           download=False,
                                           transform=train_transform)
test_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=False, download=False,
                                           transform=test_transform)

m = len(train_dataset)

train_data, val_data = random_split(dataset=train_dataset,
                                    lengths=[int(m-m*0.2), int(m*0.2)])

batch_size=256

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(dataset=val_data,
                                           batch_size=batch_size)
test_loder = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size)

# define Encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, 
                              kernel_size=3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=16)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2, padding=0)
        self.linear1 = nn.Linear(in_features=32*3*3, out_features=128)
        
        self.linear2 = nn.Linear(in_features=128, out_features=latent_dims)
        
        self.linear3 = nn.Linear(in_features=128, out_features=latent_dims)
        
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0
        
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z 
        
# define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3*3*32),
            nn.ReLU()
            )
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,3,3))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3,stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8,
                               kernel_size=3, stride=2,padding=1,
                               output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1)
            )
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
# combine the encoder and decoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
# Set the random seed for reproducible results
torch.manual_seed(0)
d = 4

vae = VariationalAutoencoder(latent_dims=d)
lr = 0.001
optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
vae.to(device)

# Training function
def train_epoch(vae, device, dataloader, optimizer):
    vae.train()
    train_loss = 0.0
    
    for x,_ in dataloader:
        x = x.to(device)
        x_hat = vae(x)
        # the loss is composed of two terms: reconstruction term +KL divergence
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'partial train loss (single batch): {loss.item()}')
        train_loss +=loss.item()
        
    return train_loss/len(dataloader.dataset)
        
        
# Testing function
     
def test_epoch(vae, device, dataloader):
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss +=loss.item()
            
    return val_loss / len(dataloader.dataset)


# Function to visualize the input and the coressponding reconstruction version

def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()        
        
# train the VAE 
num_epochs = 50

for epoch in range(num_epochs):
   train_loss = train_epoch(vae,device,train_loader,optim)
   val_loss = test_epoch(vae,device,valid_loader)
   print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   # plot_ae_outputs(vae.encoder,vae.decoder,n=10)




















