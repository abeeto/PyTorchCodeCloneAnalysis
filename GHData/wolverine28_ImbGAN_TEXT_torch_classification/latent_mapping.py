import torch
from torch.nn import Module
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np

class latent_mapping(nn.Module):
    def __init__(self, z_dim):
        super(latent_mapping, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(self.z_dim*2,self.z_dim),
            nn.BatchNorm1d(self.z_dim)
            )

    def forward(self, z):
        return self.model(z).unsqueeze(0)