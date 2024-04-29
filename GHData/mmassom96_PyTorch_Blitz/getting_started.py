import torch

# uninitialized 5x3 matrix
x = torch.empty(5, 3)
print(x)

# 5x3 matrix initialized with random numbers
x = torch.rand(5, 3)
print(x)

# 5x3 matrix initialized with zeros and of dtype long
x = torch.zeros(5, 3, dtype = torch.long)
print(x)

# a tensor constructed directly with the data [5.5, 3]
x = torch.tensor([5.5, 3])
print(x)

# tensors based on an existing tensor
x = x.new_ones(5, 3, dtype = torch.double)
print(x)

x = torch.randn_like(x, dtype = torch.float)
print(x)
# gets the size of x
print(x.size())

# another 5x3 matrix initialized with random numbers
y = torch.rand(5, 3)
# two ways to print the sum of tensor x and tensor y
print(x + y)
print(torch.add(x, y))

# a third way is to privide a tensor as an output argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# in-place addition
y.add_(x)
print(y)

# standard NumPy-like indexing
print(x[:, 1])

# resize/reshape a tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# .item() for one element tensors
x = torch.randn(1)
print(x)
print(x.item())

# converting a Torch tensor to a NumPy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# the NumPy array is changed
a.add_(1)
print(a)
print(b)

import numpy as np

# converting a NumPy array to Torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))




