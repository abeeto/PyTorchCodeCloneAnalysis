import torch as t
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, variance=0.01):
        super().__init__()
        self.variance = variance

    def forward(self, x):
        if self.training:
            return x + t.randn(x.size(), device=x.device) * self.variance
        return x
