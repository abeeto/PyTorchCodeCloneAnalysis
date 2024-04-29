# ====Tensor====
import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor operation:
y = x + 2
print(y)

# y was created as a result of an operation, so it has a grad_fn.
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
out = z.mean()

print(z, out)

# .requires_grad_(...) changes an existing Tensor's requires_grad flag in-place.
# The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# ====Gradients====

# Let's backprop now. Because out contains a single scalar,
# out.backward() is equivalent to out.backward(torch.tensor(1.)).
out.backward()

# Print gradients d(out)/dx
print(x.grad)

# Now let's take a look at an example of vector-Jacobian produck:
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# Now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly,
# but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# You can also stop autograd from tracking history on Tensors
# with .requires_grad=True by wrapping the code block in with torch.no_grad():
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)
