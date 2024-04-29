# Comparison mask.
# Source: https://stackoverflow.com/questions/58521595/masking-tensor-of-same-shape-in-pytorch

import torch


x = torch.Tensor(list(range(6))).reshape(2, 3)
print(x.dtype)    # torch.float32
mask = x <= 2

print("x:", x)
print("mask:", mask)

y = torch.Tensor([100] * 6).reshape(2, 3)
y[mask] = x[mask]
print("y:", y)

y = x.clone()
y[mask] *= 2
print("y:", y)

y = x.clone()
y[mask] += 1
print("y:", y)
