from __future__ import print_function
import torch

# Construct uninitialized matrix
x = torch.empty(5, 3)
print(x)

# Construct randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# Construct matrix filled zeros (dtype is long)
x = torch.zeros(5, 3, dtype = torch.long)
print(x)

# Construct a tensor with data
x = torch.tensor([5.5, 3])
print(x)

# Create a tensor based on and existing tensor
x = x.new_ones(5, 3, dtype = torch.double)
print(x)

x = torch.randn_like(x, dtype = torch.float)
print(x)

# Get its size
print(x.size())

# Addition operation
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out = result)
print(result)

y.add_(x)
print(y)

# Can use NumPy-like indexing
print(x[:, 1])

# Tensor resize, reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 is inferred from other dimensions (16 / 8 = 2)
print(x.size(), y.size(), z.size())

# Get the value as a python number from one element tensor
x = torch.randn(1)
print(x)
print(x.item())

# Converting a Torch tensor to a NumPy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b) # Torch tensor and NumPy array share underlying memory locations (directly refer)

# Converting a NumPy array to a Torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a)
print(b)

# Tensors can be moved onto any device
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device) # x is moved onto CUDA device
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))