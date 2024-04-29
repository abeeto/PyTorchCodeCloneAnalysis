####################################################################
# Activation Functions & Max Pooling
# 1. Applying an activation function, which is analogous to building
#    a regular network.
# 2. Max pooling reduces the number of parameters and makes the network
#    less susceptible to changes in the image.
####################################################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc


######################## Activation Function #######################
# Apply activation function after running the image through kernel
####################################################################
# Create a kernel and image as usual. Set the bias to zero
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
Gx = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])
conv.state_dict()['weight'][:][:] = Gx
conv.state_dict()['bias'][:] = 0.0

# Create an image
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1

# Apply convolution to image
Z = conv(image)
print(Z)

# Apply the activation function to the activation map
A = torch.relu(Z)
print(A)
# Create an activation function object
relu = nn.ReLU()
print(relu(Z))


############################ Max Pooling ###########################
# Max pooling simply takes the maximum value in each region.
####################################################################
# Consider the following image
image1 = torch.zeros(1, 1, 4, 4)
image1[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, -4.0])
image1[0, 0, 1, :] = torch.tensor([0.0, 2.0, -3.0, 0.0])
image1[0, 0, 2, :] = torch.tensor([0.0, 2.0, 3.0, 1.0])

# Create a maxpooling object in 2d
max1 = nn.MaxPool2d(2, stride=1)  # (kernel_size, stride)
print(max1(image1))

# stride=None: stride = kernel_size
maxpooled = torch.max_pool2d(image1, stride=None, kernel_size=2)
print(maxpooled)



