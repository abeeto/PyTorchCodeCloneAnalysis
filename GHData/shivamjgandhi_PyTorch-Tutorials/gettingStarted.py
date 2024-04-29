from __future__ import print_function
import torch

# Construct a 5x3 matrix

x = torch.empty(5, 3)
print("this is a 5x3 matrix", x)

# Contstruct a random 5x3 matrix

x = torch.rand(5, 3)
print("this is a random 5x3 matrix", x)

# Construct a matrix of zeros and dtype long

x = torch.zeros(5, 3, dtype=torch.long)
print("this is a zeros matrix of longs", x)

# Construct a tensor directly from data

x = torch.tensor([5.5, 3])
print("this is a matrix construct from data", x)

# Construct a tensor based on an existing tensor

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print("same size as last x", x)

# size of a tensor
print(x.size())

# Add tensors

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("this time, we set which variable is the set by addition", result)

# In place addition
y.add_(x)
print(y)

# Resize/reshape tensors
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# Turn a 1 element tensor into a number
x = torch.randn(1)
print(x)
print(x.item())

# Convert a Torch Tensor to a Numpy Array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Convert a numpy array to a torch tensor

import numpy as np 
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
