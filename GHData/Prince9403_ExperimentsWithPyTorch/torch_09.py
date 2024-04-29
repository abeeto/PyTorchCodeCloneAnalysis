import torch

x = torch.autograd.Variable(11 * torch.ones(1), requires_grad=True)
print("Start x value:", x)

t = torch.autograd.Variable(11 * torch.ones(1), requires_grad=True)

z = 7 * torch.randn(1)
print("Type of z:", type(z))

optimizer = torch.optim.Adam([x], lr=0.01)

for i in range(10000):
    optimizer.zero_grad()
    y = x * x - 9 * x + t
    y = t * y
    y.backward()
    optimizer.step()

print("Type of x:", type(x))
print("Final x value:", x)
