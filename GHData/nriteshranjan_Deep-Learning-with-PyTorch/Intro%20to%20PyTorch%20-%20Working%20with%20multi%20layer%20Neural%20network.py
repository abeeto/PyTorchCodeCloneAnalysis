# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:07:47 2020

@author: rrite
"""

import torch
def activation(x):
  return (1 / (1 + torch.exp(-x)))
# Generate some data
torch.manual_seed(7) #sets the random seed so that things are predictable

# Features are 3 random variable
features = torch.randn((1,3))

# Define the size of each layer in our neural network
n_input = features.shape[1] # Number of input units, must match number of input features
n_hidden = 2                # Number of hidden units
n_output = 1                # Number of output units
 
# Weights for inputs to hidden layer
W1 = torch.rand(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

#Printing above variables
print(features)
print(n_input)
print(W1)
print(W2)
print(B1)
print(B2)

# Getting output of above multi-layer
output1 = activation(torch.mm(features, W1 ) + B1)
y = activation(torch.mm(output1, W2) + B2)
print(y)
