import torch
import torch.nn as nn
import numpy as np
import os

os.system('cls')

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
outs = softmax(x)
print('softmax : ', outs)

x = torch.tensor([2.0, 1.0, 0.1])
outs = torch.softmax(x, dim = 0)
print('torch outs : ', outs)