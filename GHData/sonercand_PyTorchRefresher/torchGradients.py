import torch
import numpy as np


x = torch.tensor(1.0, requires_grad=True)
y = x**2  # 1
y.retain_grad()
z = y + y**2  # 2
z.backward()
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
