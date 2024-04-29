import torch


x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y ** 2 * 3
out = z.mean()

out.backward()


print(x.grad)
