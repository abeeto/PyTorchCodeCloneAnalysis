import torch
from torch import nn
from torch.nn import functional as F


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = F.softplus(i)
        result = i * torch.tanh(result)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sp = F.softplus(i)
        grad_sp = 1 - torch.exp(-sp)
        tsp = torch.tanh(sp)
        grad_tsp = (1 - tsp*tsp) * grad_sp
        grad = i * grad_tsp + tsp
        return grad_output * grad


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


