# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:57:32 2020

@author: rrite
"""

#Working with single layer neural network
import torch
def activation(x):
  return (1 / (1 + torch.exp(-x)))

#Generating some random number
torch.manual_seed(7)  #manual_seed : Sets the seed for generating random numbers.

#Features are 5 random normal variable
features = torch.randn((1,5)) #randn : Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1

#True weights for our data, random normal variable again
weights = torch.randn_like(features) #randn_like : Returns a tensor with the same size as input that is filled with random numbers from a normal distribution

#and a True bias term
bias = torch.randn(1, 1)

#Checking variables
print(features)
print(weights)
print(bias)

#Creating a simple neural net
y = activation(torch.sum(features * weights) + bias)
#or
y = activation((features * weights).sum() + bias)

#linear algebra operation are highly efficient due to GPU acceleration
import tensorflow as tensor
#print(tensor.shape(weights))
#weights = weights.view(5, 1)
#print(tensor.shape(weights))
y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)