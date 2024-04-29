import torch

x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x*y)

x1 = torch.zeros([2, 5])

print(x1)

print(x1.shape)

z = torch.rand([2,5])
print(z)

z1 = z.view([1, 10])
print(z1)
