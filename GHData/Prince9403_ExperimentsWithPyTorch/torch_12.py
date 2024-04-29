import torch

x = torch.autograd.Variable(7 * torch.ones(1), requires_grad=True)

y = 2 * x
z = y * y
z.backward()

t = torch.sin(y)
t.backward()
