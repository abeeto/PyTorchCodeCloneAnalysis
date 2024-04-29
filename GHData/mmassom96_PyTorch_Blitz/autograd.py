import torch

# setting a tensor to requires_grad=True to track computation
x = torch.ones(2, 2, requires_grad=True)
print(x)

# performing a tensor operation
y = x + 2
print(y)

# showing grad_fn
print(y.grad_fn)

# more operations on y
z = y * y * 3
out = z.mean()
print(z, out)

# changing a tensor's requires_grad flag in-place
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# backward operation
out.backward()
print(x.grad)

# vector-jacobian procuct
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# backward operation
v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)
print(x.grad)

# stop autograd using with
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)
    