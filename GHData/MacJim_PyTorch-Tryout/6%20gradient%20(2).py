# Source: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch


# MARK: - Change require_grad
a = torch.randn(2, 2)    # Standard normal distribution
a = ((a * 3) / (a - 1))
print(a.requires_grad)    # False

# Both method seem to work.
# a.requires_grad_(True)
a.requires_grad = True
print(a.requires_grad)

b = (a * a).sum()
print(b.item())
print(b.grad_fn)

print()


# MARK: - grad_fn
x = torch.ones(2, 2, requires_grad=True)
print("x:", x)

y = x + 2
print("y:", y)
print(y.grad_fn)    # y was created as a result of an operation, so it has a grad_fn

z = y * y * 3
print("z:", z)
out = z.mean()
print(out)
print(out.item())

print()


# MARK: - Backprop
out.backward()    # Here `out` is a scalar: 1/4 * 6 * (x + 2)
print(x.grad)


# MARK: - Backprop 2
x = torch.randn(3, requires_grad=True)
print("x:", x)
y = x ** 2    # Here y is not a scalar.
print("y:", y)
y.backward(torch.Tensor([1, 1, 1]))    # Multiply the parameter vector (`v`) with the "Jacobian matrix" (`J`). Please see source web page.
print(x.grad)

print()


# MARK: - Do not backprop 1
x = torch.randn(3, requires_grad=True)
print((x ** 2).requires_grad)    # True
with torch.no_grad():
    print((x ** 2).requires_grad)    # False

print()


# MARK: - Do not backprop 2
x = torch.randn(3, requires_grad=True)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
print(x == y)    # (x == y) seems much simpler than (x.eq(y)).

print()
