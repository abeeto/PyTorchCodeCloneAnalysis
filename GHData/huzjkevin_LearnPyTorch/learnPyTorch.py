from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5,3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5, 3])
print(x)

ax = x.new_ones(5,3,dtype=torch.double)
print(ax)

x = torch.randn_like(ax, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5,3)
print(x + y)

print(torch.add(x, y))

result = torch.add(x,y)
print(result)

y.add_(x)
print(y)

print(x[:,1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
print(x[0,0])

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

import numpy as np

print("here")

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))