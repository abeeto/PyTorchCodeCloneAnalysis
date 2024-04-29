import torch
import torch.nn as nn
import torch.nn.functional as fn

class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, -1)

class ConvMNISTVAE(nn.Module):
  def __init__(self):
    super(ConvMNISTVAE, self).__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=5),
      nn.ReLU(True),
      nn.Conv2d(8, 16, kernel_size=5),
      nn.ReLU(True),
      Flatten(),
      nn.Linear(6400, 400)
    )
    self.mu_layer = nn.Linear(400, 20)
    self.logvar_layer = nn.Linear(400, 20)
    self.fully_connected_layers = nn.Sequential(
      nn.Linear(20, 400),
      nn.ReLU(True),
      nn.Linear(400, 6400)
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(16, 8, kernel_size=5),
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, kernel_size=5),
      nn.Sigmoid()
    )

  def __reparam__(self, mu, logvar):
    std = 0.5 * torch.exp(logvar)
    epsilon = torch.rand_like(std)
    return mu + std * epsilon

  def encode(self, x):
    x = self.encoder(x)
    return self.mu_layer(x), self.logvar_layer(x)

  def decode(self, x):
    x = self.fully_connected_layers(x)
    x = x.view(-1, 16, 20, 20)
    x = self.decoder(x)
    return x

  def forward(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    z = self.__reparam__(mu, logvar)
    x = self.fully_connected_layers(z)
    x = x.view(-1, 16, 20, 20)
    x = self.decoder(x)
    return x, mu, logvar
