#! /usr/bin/python

# This is an example code for getting started with PyTorch
print("getting started with PyTorth\n")

import torch
import numpy as np

# Tensors
x1 = torch.Tensor(5,3) # Construct a 5x3 matrix, uninitialized
print("x1:"); print(x1)
x2 = torch.rand(5,3) # Construct a randomly initialized matrix
print("x2:"); print(x2)
x3_ = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
x3 = torch.Tensor(x3_) # Construct matrix initialized by ndarray
print("x3:"); print(x3)
print(x3.size())
# torch.Size is in fact a tuple, so it supports the same operations

# Operations
# Addition: syntax 1
y1 = torch.add(x2,x3)
print("y1:"); print(y1)
# Addition: syntax 2
y2 = x2 + x3
print("y2:"); print(y2)
# Addition: syntax 3
y3 = torch.Tensor(5,3)
torch.add(x2,x3,out=y3)
print("y3:"); print(y3)
# Addition: in place
y4 = x2.clone()
y4.add_(x3)
print("y4:"); print(y4)
# standard numpy-like indexing is supported
y5 = x3[0,:]
print("y5:"); print(y5)
# expand/squeeze dimensions
x3_hat = x3.clone()
x3_hat = torch.unsqueeze(x3_hat, 0) # [5,3] ----> [1,5,3]
print(x3_hat.size())
x3_hat = torch.squeeze(x3_hat) # [1,5,3] ----> [5,3]
print(x3_hat.size())

# Numpy Bridge
# from Tensor to ndarray
a = torch.ones(3)
print("a:"); print(a)
b = a.numpy()
print("b:"); print(b)
a.add_(1)
print("b_new:"); print(b) # the value of ndarray changes according to torch Tensor
# from ndarray to Tensor
c = torch.from_numpy(b)
print("c:"); print(c)
a.add_(1)
print("c_new:"); print(c) # the values change in a chain: a->b->c

# CUDA Tensors
# Tensors can be moved onto GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x2 = x2.to(device)
    x3 = x3.to(device)
    y = x2 + x3
