"""
Linear layer with weights sampled from a distribution.
this linear layer implementation is inspired from the https://github.com/mjpyeon/pytorch-bayes-by-backprop repo. Credits to the owner.
"""

#TODO try with other distributions, not just gaussian distribution


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def log_gaussian_prob(x, mu, sigma, log_sigma=False):
    if not log_sigma:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - torch.log(sigma) - 0.5*(x-mu)**2 / sigma**2
    else:
        element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - F.softplus(sigma) - 0.5*(x-mu)**2 / F.softplus(sigma)**2
    return element_wise_log_prob.sum()



class GaussianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, stddev_prior=0.003,bias=True):
        super(GaussianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stddev_prior = stddev_prior
        self.w_mu = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior))  # local reparamatrization trick
        self.w_rho = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior)) #local reparametrization trick
        self.b_mu = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
        self.b_rho = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
        self.bias = bias
        self.q_w = 0.
        self.p_w = 0.

    def forward(self, x, test=False):
        if test:
            w = self.w_mu
            b = self.b_mu if self.bias else None
        else:
            device = self.w_mu.device
            w_stddev = F.softplus(self.w_rho)
            b_stddev = F.softplus(self.b_rho) if self.bias else None
            w = self.w_mu + w_stddev * torch.Tensor(self.in_dim, self.out_dim).to(device).normal_(0,self.stddev_prior)
            b = self.b_mu + b_stddev * torch.Tensor(self.out_dim).to(device).normal_(0,self.stddev_prior) if self.bias else None
            self.q_w = log_gaussian_prob(w, self.w_mu, self.w_rho, log_sigma=True)
            self.p_w = log_gaussian_prob(w, torch.zeros_like(self.w_mu, device=device), self.stddev_prior*torch.ones_like(w_stddev, device=device))
            if self.bias:
                self.q_w += log_gaussian_prob(b, self.b_mu, self.b_rho, log_sigma=True)
                self.p_w += log_gaussian_prob(b, torch.zeros_like(self.b_mu, device=device), self.stddev_prior*torch.ones_like(b_stddev, device=device))
        output = x@w+b
        return output

    def get_pw(self):
        return self.p_w

    def get_qw(self):
        return self.q_w





