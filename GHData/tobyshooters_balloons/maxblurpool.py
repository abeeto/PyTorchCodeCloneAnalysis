import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxBlurPool(torch.nn.Module):
    """
    Simplified implementation of MaxBlurPool based on Adobe's antialiased CNNs.
    """
    def __init__(self, n):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 1)
        self.padding = nn.ReflectionPad2d(1)

        f = torch.tensor([1, 2, 1])
        f = (f[None, :] * f[:, None]).float()
        f /= f.sum()
        f = f[None, None].repeat((n, 1, 1, 1))

        self.register_buffer('f', f)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.padding(x)
        x = F.conv2d(x, self.f, stride=2, groups=x.shape[1])
        return x
