import torch
import numpy as np

# Construct a 5x3 matrix, uninitialized:
x = torch.empty(5, 3)
print("x:", x)

# Construct a randomly initialized matrix:
x1 = torch.rand(5, 3)
print("x1:", x1)

# Construct a matrix filled zeros and of dtype long:
x0 = torch.zeros(5, 3, dtype=torch.long)
print("x0:", x0)
a = torch.tensor([5.5, 3])
print("a:", a)

#  create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
a_new = a.new_ones(5, 3, dtype=torch.double)
print("a_new:", a_new)
a_new1 = torch.randn_like(a_new, dtype=torch.float)
print(a_new1)
print(a_new1.size())

# Operations
print(a_new1 + x1)
print(torch.add(a_new1, x1))
# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(a_new1, x1, out=result)
print(result)

# Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
# adds a_new1 to x
print("adds a_new1 to x:", x1.add_(a_new1))

# Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print("Resizing:", x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print("item():", x.item())

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)
b = a.numpy()
print("convert to numpy：", b)
# See how the numpy array changed in value.At the same time
a.add_(1)
print(a, b)

# Converting NumPy Array to Torch Tensor
print("Converting NumPy Array to Torch Tensor")
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
