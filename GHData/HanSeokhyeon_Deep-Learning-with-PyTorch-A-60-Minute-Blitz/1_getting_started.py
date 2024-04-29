# Getting Started


# ====Tensors====

from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x)

# or create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype,
# unless new values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# Get its size:
print(x.size())


# ====Operations====

# Addition: syntax 1
y = torch.rand(5, 3)
print(x + y)

# Addition: syntax 2
print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
y.add_(x)
print(y)

# You can use standard Numpy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the values as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# ====NumPy Bridge====

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting Numpy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# ====CUDA Tensors====

# Tensors can be moved onto any device using the .to method.
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
