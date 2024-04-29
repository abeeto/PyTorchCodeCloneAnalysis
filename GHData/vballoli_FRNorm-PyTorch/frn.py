import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#__all__ = ['fr_norm', 'FRNorm']

def fr_norm(tau, beta, gamma, eps, x):
    """
    Functional implementation of the Filter Response Normalization layer.

    Computes the mean on the input activations and performs FRN using the input and the mean.
    """
    assert len(x.size()) == 4, "Input dimensions must be 4D, but given " + str(len(x.size()))
    # Computes the mean of input activations along each channel.
    nu2 = torch.mean(x, axis=[2, 3], keepdims=True)
    # Computes the FRN activation using the input and it's mean
    x = x * torch.rsqrt(nu2 + eps)
    # Applies Offset-ReLU non-linearity
    return torch.max(gamma * x + beta, tau)


class FRNorm(nn.Module):
    """
    Filter Response Normalization layer from the paper by Singh et al.
    Link to the paper - https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, num_features, eps=1e-6, *args):
        """
        Arguments -
        1. num_features - tuple containing (B, C, H, W)
        """
        super(FRNorm, self).__init__()
        assert len(num_features) == 4
        self.tau = Variable(torch.randn(*num_features), requires_grad=True)
        self.beta = Variable(torch.randn(*num_features), requires_grad=True)
        self.gamma = Variable(torch.randn(*num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Computes the mean on the input activations and performs FRN using the input and the mean.

        X must be of format BxCxHxW
        """
        return fr_norm(self.tau, self.beta, self.gamma, self.eps, x)