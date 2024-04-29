# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:22:10 2020

@author: NjordSoevik
"""

import torch # Tensor Package (for use on GPU)
import torch.optim as optim # Optimization package
import matplotlib.pyplot as plt # for plotting
import numpy as np
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
from torch.autograd import Variable # for computational graphs

# this block of code is organized a little differently than section 5, but it's mostly the same code
# the only three differences are:
# - The "Hyperparameter" constants
# - The for loop (for helping the model do <number of epochs> training steps)
# - The linear_layer1.zero_grad() function call on line 25. 
#   (that's just to clear the gradients in memory, since we're starting the training over each iteration/epoch)

# input
x1_var=Variable(torch.Tensor([1,2,3,4]),requires_grad=True)
# layer with input output dimensions
linear_layer1=nn.Linear(4,1)
# output
target_y=Variable(torch.Tensor([0]),requires_grad=False)
# predict x into layer
print("----------------------------------------")
print("Output (BEFORE UPDATE):")
print(linear_layer1(x1_var))

epochs=3
learning_rate=1e-4
loss_f=nn.MSELoss()
optimizer=optim.SGD(linear_layer1.parameters(),lr=learning_rate)

for e in range(epochs):
    # zero out gradients
    linear_layer1.zero_grad()
    # predict
    predicted_y=linear_layer1(x1_var)
    # calculate loss of predict to real
    loss=loss_f(predicted_y,target_y)
    # calc gradients w/ loss
    loss.backward()
    # optimizer to minus grad * lr from weights
    optimizer.step()
    print("----------------------------------------")
    print("Output (UPDATE " + str(epochs + 1) + "):")
    print(linear_layer1(x1_var))
    print("Should be getting closer to 0...")

print("----------------------------------------")