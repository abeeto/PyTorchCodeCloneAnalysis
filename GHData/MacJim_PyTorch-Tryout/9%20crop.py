import torch


x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = x.view(2, -1)
print(x)

y = x[:, 1:3]
print(y)
