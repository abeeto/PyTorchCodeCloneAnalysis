import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def uniform_dist(a,b,size):
    std_unif = torch.rand(size)
    return std_unif*(b-a)+a

def safe_log(x):
    return torch.log(x+1e-4)