import torch

#set requires_grad to track computation
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z,out)

#change the requires_grad_ flag in place
a = torch.randn(2,2)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

#GRADIENTS
#if you do the backward without any gradient, the default one is a
#torch.tensor(1.)
out.backward()
print(x.grad)

#crazy thing made possible with autograd

x = torch.randn(3,requires_grad = True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(gradients)

print(x.grad)

with torch.no_grad():
    print(x.requires_grad)