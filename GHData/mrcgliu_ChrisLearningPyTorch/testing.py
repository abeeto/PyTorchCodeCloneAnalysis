import torch

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

y1 = rand_tensor @ zeros_tensor.T

y2 = rand_tensor.matmul(zeros_tensor.T)

y3 = torch.rand_like(rand_tensor)
torch.matmul(rand_tensor, zeros_tensor.T, out=y3)

print(y1)
print(y2)
print(y3)