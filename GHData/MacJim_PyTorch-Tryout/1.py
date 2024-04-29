# Source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
import torch


# MARK: Tensor creation
# A tensor is a matrix.
# Empty tensor with unpredictable values.
x = torch.empty(5, 3)
print(type(x))    # torch.Tensor
print(x.dtype)
print(x.size())    # 5, 3
print("Empty tensor:", x)

x = torch.rand(5, 3)
print(x)

# Normal distribution with mean 0 and variance 1, aka standard normal distribution.
x = torch.randn(5, 3)
print("Standard normal distribution:", x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Tensor with values.
x = torch.Tensor([[2, 3], [4, 5]])
print(x)

x = torch.ones(5, 3)
print(x)

x = torch.ones(5)
print(x)
print(x.size())

x = torch.eye(5)    # Yes! A single parameter is enough.
print(x)


# MARK: One element tensor
x = torch.ones(1, 1)
print(x)
print(x.item())    # Get the single value.

# x = torch.ones(2, 2)
# print(x.item())    # This leads to a value error.
