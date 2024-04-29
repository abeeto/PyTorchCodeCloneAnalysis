import torch

x= torch.randn(3, requires_grad=True)
print(x)
y=x+2
print(y)