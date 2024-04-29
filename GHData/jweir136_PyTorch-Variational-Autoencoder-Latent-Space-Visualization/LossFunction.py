import torch
import torch.nn as nn
import torch.nn.functional as fn

def loss_function(pred_x, x, mu, logvar):
  bce = fn.binary_cross_entropy(pred_x, x)
  kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
  return bce + kld
