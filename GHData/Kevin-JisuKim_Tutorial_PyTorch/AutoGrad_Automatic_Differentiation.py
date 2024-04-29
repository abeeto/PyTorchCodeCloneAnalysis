import torch

# Create a tensor and track computation
x = torch.ones(2, 2, requires_grad = True)
print(x)

# Addition operation
y = x + 2
print(y)

# Grad_fn is AddBackward because y is a result of an addition operation
print(y.grad_fn)

# Grad_fn is MulBackward and MeanBackward because z is a result of an multiplication and mean operation
z = y * y * 3
out = z.mean()

print(z, out)

# Change an existing tensor's requires_grad flag (default is false)
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# Backpropagation
out.backward()

print(x.grad)

# Vector-Jacobian product (differentiation vector)
x = torch.randn(3, requires_grad = True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
y.backward(v)

print(x.grad)

# Can stop tracking history
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())