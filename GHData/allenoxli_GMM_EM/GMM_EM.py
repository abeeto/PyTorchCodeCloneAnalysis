import math
import numpy as np
import torch

from plot_clusters import PlotAllClusters


class GMM:
    def __init__(self, n: int, d: int, k_components: int):
        r"""Construct and initialize parameters.

        args:
            n: Numbers of input data x.
            d: Numbers of x's dimension.
            k_components: Numbers of gaussian distribution.
        attritubes:
            alpha:
                Probability of selecting each gaussian component.
                size(1, k, 1)
            mu:
                The mean of gaussian indicates distribution's center location.
                size(1, k, d)
            var:
                The variance of the gaussian indicates distribution's width.
                size(1, k, d)
            gamma:
                Probability of each x belong to each component.
                size(n, k, 1)
        """
        self.n = n
        self.d = d
        self.k_components = k_components

        self.alpha = torch.ones(1, k_components, 1).fill_(1./k_components)
        self.mu = torch.rand(1, k_components, d)
        self.var = torch.ones(1, k_components, d)

        self.gamma = torch.ones((n, k_components)) / k_components


    def train(self, x):
        log_likelihood = self.estimate_log_likelihood(x)
        self.e_step(x)
        self.m_step(x)

        return self.mu, self.var, log_likelihood


    def gaussian_function(self, x):
        r"""Return tensor means probability of each x belong to each gaussians component.

        args:
            x: size(n, 1, d)
        return:
            prob: size(n, k, 1)
        """
        mean = self.mu
        std = np.sqrt(self.var)
        pi = torch.tensor(math.pi)
        d = self.d

        std_mul = std.log().sum(dim=-1, keepdim=True).exp()
        divisor_part = 1 / (std_mul * np.sqrt((2*pi))**d)

        exp_part = (-0.5) * ((x-mean)/std)**2
        exp_part = exp_part.sum(dim=-1, keepdim=True).exp()

        return divisor_part * exp_part


    def e_step(self,x):
        r"""According to posterior probability, we can compute x belonging to which gaussian component.
            return:
                size:(n, k, 1)
        """
        # (n, k, 1)
        per_x_prob_table = self.alpha * self.gaussian_function(x)
        # (n, 1, 1)
        per_x_prob_sum = per_x_prob_table.sum(dim=1,  keepdim=True)
        # (n, k, 1)
        self.gamma = per_x_prob_table / per_x_prob_sum

    def m_step(self, x):
        r"""Update model's parameters.

        Update parameters:
            alpha:
                size(1, k, 1)
            mu:
                size(1, k, d)
            var:
                size(1, k, d)
        """
        # size(1, k, 1)
        n_k = self.gamma.sum(dim=0, keepdim=True)

        # Update parameters.
        self.alpha = n_k / self.n
        self.mu = torch.sum((self.gamma * x), dim=0, keepdim=True) / n_k
        self.var = torch.sum((self.gamma * (x - self.mu)**2), dim=0, keepdim=True) / n_k

    def estimate_log_likelihood(self, x):
        r"""Estimating log likelihood.
        log_likelihood = Sum(log(P(x|model)))
        retrun:
            size(1)
        """
        # sum(x1~xn): alpha_k * N(x| mu, std)
        # size: (1)
        per_x_prob_table = self.alpha * self.gaussian_function(x)
        per_x_log_prob_sum = torch.log(per_x_prob_table.sum(dim=1))

        return torch.sum(per_x_log_prob_sum)

    def predict(self, x):
        r"""Predict x belong to which gaussian component."""
        per_x_prob_table = self.alpha * self.gaussian_function(x)

        _, idx = torch.max(per_x_prob_table, dim=1)

        return idx


