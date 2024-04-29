import numpy as np
import torch

# indexing, slicing, reshaping #######

x = torch.arange(9).reshape(shape=(3, 3))
print(x)

# index
print(x[2, 1])
# slice - get second column
print(x[:, 1], x[:, 1].shape)  # returns a row
# slice - get second row
print(x[1, :], x[1, :].shape)  # returns a row
# slice by keeping the column shape
print(x[:, 0:1])  # first column
print(x[:, 1:2])  # get second column
print(x[:, 2:3], x[:, 2:3].shape)  # get the third column

# reshape and view ##################
x = torch.arange(9)
print(x)
y = x.reshape(3, 3)
print(y)
y2 = x.view(3, 3)
print(y2)
x[0] = 999
print(y, y2, x)
# reshape by infering one of the dimentions###
x = torch.arange(10)
print(x.view(2, -1))  # 2 row and column number is infered
print(x.view(-1, 2))

### Arithmetic #########
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([0, 1, 2], dtype=torch.float32)
print(a + b)
print(torch.add(a, b))
print(a.mul(b))
print(a)
print(
    a.mul_(b)
)  # multiply inplace=True  # NOTE: underscore operator means inplace=True
print(a)

### Matrix Ops #########
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([2.0, 3.0, 4.0])
print(a * b)  # elementwise multiplication == mult(a*b)
print(a.dot(b))  # dot product

a = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])  # 2by3
b = torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])  # 3by2
print(torch.mm(a, b))  # matrix multiplication == a@b

# ecludian norm
print(a.norm())
# totam num of elements in a tensor
print(a.numel())
