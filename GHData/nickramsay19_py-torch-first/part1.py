import torch

x = torch.Tensor([2,5])
print(x)

y = torch.rand([2,5])
print(y.shape)

# flatten or reshape
y.view([1,10])
print(y)

