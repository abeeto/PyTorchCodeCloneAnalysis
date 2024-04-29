# Source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch


# MARK: Operations
x = torch.ones(5, 3)
y = torch.ones(5, 3)
z = x + y
print(z)

y += x    # It works.
print(y)

y += 2    # It works! Every element got increased by 2.
print(y)

x = torch.Tensor([[1, 2], [3, 4]])
y = torch.ones(2, 2)
z = x * y
print("Element multiply:", z)    # This is element wise multiply.
z = x @ y
print("Matrix multiply:", z)


# MARK: Batch operation
x = torch.Tensor(list(range(4)))
print("Before batch add:", x)
x += 1
print("After batch add:", x)
x *= 2
print("After batch mul:", x)


# MARK: Size
x = torch.ones(5, 3)
y = x[:, 1]
print(x.size())
print(x)
print(y.size())
print(y)


# MARK: Reshape
x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
z1 = x.view(2, 4)
z2 = x.view(-1, 4)    # -1 means calculate automatically.

print(z1.size())
print(z1)
print(z2.size())
print(z2)


x = torch.Tensor(list(range(8)))
y = x.view(2, 1, 2, 2)
z = y.view(2, 2, 2)

print(y.size())
print(y)
print(z.size())
print(z)


# MARK: Sum
x = torch.Tensor(list(range(20)))
x = x.reshape(4, -1)
print("x:", x)

y = torch.sum(x, dim=0)
z = torch.sum(x, dim=1)
print("Dim 0 sum:", y)
print("Dim 1 sum:", z)


# MARK: Power
x = torch.Tensor(list(range(4)))
x = x.reshape(2, 2)
y = torch.pow(x, 2)
print("y:", y)
