import torch

x= torch.randn(3, requires_grad=True)
print(x)
x.requires_grad_(False)
print(x)