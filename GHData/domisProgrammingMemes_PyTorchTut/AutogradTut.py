# second Tutorial - Autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
# torch.autograd is an engine for computing vector-Jacobian product

import torch
# create Tensor and set 'require_grad = True' to track the computation
# tensor 1x4 bedeuted: 1 Reihe, 4 Spalten (1 row 4 columns)
print("Tensors and Autograd:")
x = torch.ones(2, 2, requires_grad=True)
print(x)

# tensor operation
y = x + 2
print(y)

# y was created as a result of an operation, so it has a 'grad_fn'
print(y.grad_fn)

# more operations on y
z = y * y * 3
out = z.mean()

print(z, out)

# '.requires_grad(...)' changes an existing tensor's require_grad flag in-place. the input  flag defaults to False if not given
print("Change existing Tensor's require_grad in place with '.requires_grad(...)':")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad, ", false by default!")
a.requires_grad_(True)
print(a.requires_grad, ", changed with function!")
b = (a * a).sum()
print(b.grad_fn)

# Gradients
print()
print("Gradients:")
# out contrains a single scalar, out.backward() is equicalent to out.backward(torch.tensor(1.))
print(out, " this is out")
out.backward()
# print gradients d(out)/dx
print(x, " this is x")
print(x.grad, " this is x.grad which is d(out)/dx (dt: Ableitung nach x)")

# example of vector-Jacobian product:
print()
print("Example of vector-Jacobian product:")
x = torch.randn(3, requires_grad=True)
print(x, " this is x")
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y, " this is y")
# x.grad = None
# print(x.grad, " this is x.grad which is None")
# y is no longer a scalar. 'torch_autograd' could not compute the full jacobian directly, but
# if we just want the vector-Jacobian product, simplay pass the vector to 'backward' as argument:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad, " this is x.grad")

# stop autograd from tracking history on Tensors with .requires_grad = True' either by wrapping the code in
# with torch.no_grad():
print()
print("stop tracking Tensor with 'with torch.no_grad()':")
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print("or using '.detach()' to get a new Tensor with the same content but which does not require gradients:")
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())