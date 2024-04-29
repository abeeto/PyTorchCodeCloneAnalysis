from __future__ import print_function
import torch
import numpy as np

# Construct a 5x3 matrix, uninitialized:

x = torch.empty(5, 3)
#print(x) # prints a matrix with values that are allocated in memory at the time

# Construct a randomly initialized matrix:

x = torch.rand(5, 3)
#print(x) # randonly initialized

# Construct a matrix filled zeros of datatype long:

x = torch.zeros(5, 3, dtype=torch.long)
# print(x) #matrix of zeros

# Construct a tensor directly from data: Tensors are numpy n dimention arrays

x = torch.tensor([5.5, 3])
# print(x) #prints a tensor values 5.5 and 3

# Create a tensor based on an existing tensor

# new_* methods take in sizes
x = x.new_ones(5, 3, dtype=torch.double)      
#print(x) # matrix of ones 

# override data type!
x = torch.randn_like(x, dtype=torch.float)    
#print(x)   # result has the same size

# Get Size: 
#print(x.size()) # In the case of x (5,3)

# Addition with different syntax:

y = torch.rand(5, 3)
# print(x + y)

# print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
# print(result)

y.add_(x)
# print(y)

# Numpy Indexing:
#print(x)
#print(x[:, 1]) # keeps middle column

# Using torch.view to resize or reshape tensor
x = torch.randn(4, 4)
y = x.view(16)
#print(y) # y gets flattened into one dimension
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(z) # reshaped into 2x8
#print(x.size(), y.size(), z.size())

# if you have a one element tensor use .item() to get the value as a python number
x = torch.randn(1)
#print(x)
#print(x.item())

# Converting a Torch Tensor to a NumPy Array

a = torch.ones(5)
#print(a) # tensor 

b = a.numpy()
#print(b) # numpy array

# Add one: 

a.add_(1)
#print(a)
#print(b)

# Converting NumPy Array to Torch Tensor

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
#print(a)
#print(b)


