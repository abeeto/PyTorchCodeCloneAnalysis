import torch
from torch import pow, nn, sigmoid
import numpy as np
import random
import torch.nn.functional as F


class AReLu(nn.Module):

    def __init__(self, k=1, n=1):
        super().__init__()
        self.k = k
        self.n = n

    def forward(self, x):
        return F.relu(self.k * pow(x, self.n))


class SBAF(nn.Module):

    def __init__(self, alpha=1, k=1):
        super().__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, x):
        return 1 / 1 + self.k * pow(x, self.alpha) * pow((1 - x), 1 - self.alpha)


class Parabola(nn.Module):
    alpha = round(random.uniform(0.60, 0.65), 2)
    beta = round(random.uniform(0.48, 0.52), 2)

    def __init__(self, alpha=alpha, beta=beta):
        super().__init__()
        self.a = alpha
        self.b = beta

    def forward(self, x):
        return sum(self.a * pow(x, 2), self.b * x)


class AbsoluteValue(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Swish(nn.Module):

    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope * nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * sigmoid(x)
