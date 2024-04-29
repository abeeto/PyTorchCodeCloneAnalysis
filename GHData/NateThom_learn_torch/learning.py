import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([.1, 1.0, .0001], dtype=torch.float)
y.backward(v)

print(x.grad)