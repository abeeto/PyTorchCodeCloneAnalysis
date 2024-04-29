import numpy as np
import torch
import math

from GMM_EM import GMM

class GMM_log(GMM):
    def __init__(self, n, d, k_components):
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
        super().__init__(
            n=n,
            d=d,
            k_components=k_components
        )

    def gaussian_function(self, x):
        r"""Return tensor means probability of each x belong to each gaussians component.

        args:
            x: size(n, 1, d)
        return:
            prob: size(n, k, 1)
        """
        mean = self.mu
        var = self.var
        pi = torch.tensor(math.pi)
        d = self.d

        log_pi_part = d * torch.log(2*pi)
        log_var_part = torch.log(var).sum(dim=-1, keepdim=True)
        log_exp_part = ((x - mean)**2 / var).sum(dim=-1, keepdim=True)  # (n, k, d)


        return (-0.5) * (log_pi_part + log_var_part + log_exp_part)

    def e_step(self,x):
        r"""According to posterior probability, we can compute x belonging to which gaussian component.
            return:
                size:(n, k, 1)
        """
        # (n, k, 1)
        per_x_prob_table = torch.log(self.alpha) + self.gaussian_function(x)
        # (n, 1, 1)
        per_x_prob_sum = per_x_prob_table.logsumexp(dim=1, keepdim=True)
        # (n, k, 1)
        self.gamma = torch.exp(per_x_prob_table - per_x_prob_sum)


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
        n_k = self.gamma.sum(dim=0, keepdim=True)   # size:(1, k, 1)
        eps = torch.tensor([1e-6])

        # Update parameters.
        self.alpha = n_k / self.n
        self.mu = (self.gamma * x).sum(dim=0, keepdim=True) / n_k
        self.var = torch.sum((self.gamma * (x - self.mu)**2), dim=0, keepdim=True)/n_k


    def estimate_log_likelihood(self, x):
        r"""Estimating log likelihood.
        log_likelihood = Sum(log(P(x|model)))
        retrun:
            size(1)
        """
        # sum(x1~xn): alpha_k * N(x| mu, std)
        # size: (1)
        return torch.sum((torch.log(self.alpha) + self.gaussian_function(x)))

    def predict(self, x):
        r"""Predict x belong to which gaussian component."""
        per_x_prob_table = torch.log(self.alpha) + self.gaussian_function(x)

        _, idx = torch.max(per_x_prob_table, dim=1)

        return idx
