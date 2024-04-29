from __future__ import print_function
import torch
import numpy as np

# Getting started
# Tensor

x = torch.Tensor(5, 3)
print(x)

print(x.size())

# Operation

y = torch.rand(5, 3)
print(x + y)
# Both are the same ^v
print(torch.add(x, y))

# Output sensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

print(x[:, 1])

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
