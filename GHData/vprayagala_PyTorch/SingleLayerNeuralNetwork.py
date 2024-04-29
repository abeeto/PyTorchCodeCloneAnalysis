# -*- coding: utf-8 -*-
"""
    Single Layer Neural network in PyTorch
    This is an untrained model to calculate score of output
"""
#%%
import numpy as np
import torch
#%%
def sigmoid_activation(x):
    """
        Sigmoid Activation Function for a Neuron
        
        Input: x - Torch Tensor
    """
    return (1/(1+np.exp(-x)))
#%%
#Generate Some Random Data
torch.manual_seed(7)

x = torch.randn((1,5))
print("Feature Values: %s"%(x))

W=torch.randn_like(x)
print("Weights:%s"%(W))

b=torch.randn((1,1))
print("Bias:%s"%(b))
#%%
#Simple Network - y=sigmoid(wx+b)
y=sigmoid_activation(torch.sum(W*x) + b)
print(y)


