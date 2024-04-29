# Tensors are similar to numpy's ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
x = torch.rand(5, 3)

print(x)

print(x.size())

y = torch.rand(5, 3)

# addition: syntax 1
print(x + y)

# addition: syntax 2
print(torch.add(x, y))

# addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

## addition: in-place
y.add_(x)
print(y)  # adds x to y

print(x[:, 1])
