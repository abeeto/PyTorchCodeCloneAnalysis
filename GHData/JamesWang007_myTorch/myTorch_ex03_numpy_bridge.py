# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:03:00 2018

@author: bejin
"""

from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.empty(5,3)
print (x)

# random matrix:
x = torch.rand(5,3)
print(x)


# zeros and dtype long:
x = torch.zeros(5, 3, dtype = torch.long)
print(x)


# a tensor directly from data:
x = torch.tensor([5, 5, 5])
print(x)

x = torch.randn_like(x, dtype= torch.float)
print(x)


# git its size, x.Size is a tuple
print(x.size())


# Addition: syntax 1
x = torch.rand(3)
y = torch.rand(5,3)
print(x)
print(y)
print(x + y)

# Addition: syntax 2
print(torch.add(x,y))


# providing an output tensor as argument
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)


# Addition: in - place
# adds x to y
y.add_(x)
print(y)


'''
Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
'''

print(y[:, 1])


# torch.view
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8) # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


# use .item()
x = torch.randn(1)
print(x)
print(x.item())


# torch Tensors
torch.tensor([1.2, 3]).dtype
torch.set_default_dtype(torch.float64)
torch.tensor([1.2, 3]).dtype


# NumPy Bridge
a = torch.ones(5)
print(a)


# convert to a NumPy Array
b = a.numpy()       # .numpy()
print(b)


# a and b are sharing their underlying memory locations.
b += 1
print(a)

a.add_(1)
print(a)
print(b)


# Converting NumPy Array to Torch Tensor
# automatically
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # .from_numpy(obj)
np.add(a, 1, out=a)
print(a)
print(b)


# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

# CUDA Tensors
# Tensors can be moved onto any device using the .to method.


# let us run this cell if CUDA is availabe
# We will use "torch.device" objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")           # a CUDA device object
    y = torch.ones_like(x, device=device)   # directly create a tensor on GPU
    x = x.to(device)                        # or just use strings ".to("cuda")"
                                            # x.to("cuda")
    z = x + y                               
    print(z)    
    print(z.to("cpu", torch.double))        # ".to" can also change dtype together!
    
    

































