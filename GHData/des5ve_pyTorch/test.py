from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pandas import DataFrame




inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

#@ represents matrix multiplication in PyTorch, and the .t method returns the transpose of a tensor.
#The matrix obtained by passing the input data into the model is a set of predictions for the target variables.
def model(x):
    return x @ w.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# Train for 100 epochs
for i in range(100):
    # Generate predictions
    preds = model(inputs)
    #Computer Loss
    loss = mse(preds, targets)
    #Computer Gradients
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

print (w)
print (b)
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
