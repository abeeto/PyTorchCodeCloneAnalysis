import torch

x = torch.empty(1) #scalar value
print(x)

x = torch.empty(2,3) #2D
print(x)

x = torch.rand(2,2)
print(x)

x = torch.zeros(2,2)
print(x)

x = torch.ones(2,2, dtype=torch.int32)
print(x.size())

# Create Tensor from list
x = torch.tensor([2,3,4,5,6])
print(x.size())
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)

print(torch.add(x,y))
print(y)
y.add_(x) #inplace operation. _ specifies that
print(y)

# torch.sub()
# torch.mul()
# applicable for inplace

# get specific tesnor value
x = torch.rand(5, 3)
print(x[1,1])
print(x[1,1].item())

# reshape
x = torch.rand(4,4)
y = x.view(-1, 1)
print(y)

# from numpy
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
print(b.numpy())