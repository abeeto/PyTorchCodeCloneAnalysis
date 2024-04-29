import torch
import torch.nn as nn

class BasicMNISTVAE(nn.Module):
  def __init__(self):
    super(BasicMNISTVAE, self).__init__()

    self.encoder = nn.Sequential(
      nn.Linear(784, 400),
      nn.ReLU(True)
    )
    self.mu_layer = nn.Linear(400, 60)
    self.logvar_layer = nn.Linear(400, 60)
    self.decoder = nn.Sequential(
      nn.Linear(60, 400),
      nn.Linear(400, 784),
      nn.Sigmoid()
    )

  def __reparam__(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    epilson = torch.rand_like(std)
    return std * epilson + mu

  def encode(self, x):
    x = self.encoder(x)
    mu = self.mu_layer(x)
    logvar = self.logvar_layer(x)
    return mu, logvar

  def decode(self, x):
    x = self.decoder(x)
    return x

  def forward(self, x):
    x = self.encoder(x)
    mu, logvar = self.mu_layer(x), self.logvar_layer(x)
    z = self.__reparam__(mu, logvar)
    x = self.decode(z)
    return x, mu, logvar
