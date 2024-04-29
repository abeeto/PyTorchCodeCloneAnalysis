import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

def loss_function(x_pred, x, mu, logvar):
  bce = fn.binary_cross_entropy(x_pred, x)
  kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return bce + kld
