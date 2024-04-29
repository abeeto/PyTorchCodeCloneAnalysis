# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:10:30 2020

@author: rrite
"""
# Importing necessary packages
from IPython import get_ipython
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np
import torch
import math
import helper
import matplotlib.pyplot as plt
import tensorflow as tf

def activation(x):
  return (1 / (1 + torch.exp(-x)))

from torchvision import datasets, transforms

# Define a transform to normalise the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the Training Data
trainset = datasets.MNIST('MNIST_data/',train = True, transform = transform, download = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape) 

plt.imshow(images[1].numpy().squeeze(), cmap = 'Greys_r')

#  Solution
input = images.view(images.shape[0], - 1)
""" # https://discuss.pytorch.org/t/what-is-the-difference-of-flatten-and-view-1-in-pytorch/51790
 # or you can do this too
 features = torch.flatten(images, start_dim=1)
 # and use features instead of input
 print("Checking equality of 2 tensors : ",torch.all(input.eq(features)))
"""

n_input  = 784
n_hidden = 256             # Number of hidden units
n_output = 10              # Number of output units

# Generate some data
torch.manual_seed(7) #sets the random seed so that things are predictable
# Weights for inputs to hidden layer
W1 = torch.rand(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
b1 = torch.randn(256) 
b2 = torch.randn(10)
h = activation(torch.mm(input, W1 ) + b1)
out = torch.mm(h, W2) + b2

""" #or you can do this too
# Getting output of above multi-layer
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
output1 = activation(torch.mm(input, W1 ) + B1)
y = torch.mm(output1, W2) + B2
print(y.shape," ",out.shape)
print(torch.all(y.eq(out)))
"""

# Implementing softmax function
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim = 1).view(64,1)

probabilites = softmax(out)

print(probabilites.shape)
print(probabilites.sum(dim = 1))    