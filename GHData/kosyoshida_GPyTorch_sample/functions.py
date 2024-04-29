import math
import torch

def myfunc001(x):
    """
    one-dimensional functions
    """
    y = x*torch.sin(2*math.pi*x**2) + 0.1*torch.randn(x.size())
    return y
