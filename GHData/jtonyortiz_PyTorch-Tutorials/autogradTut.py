import torch

#Create a tensor and set requires_grad = True to track computation with it
print("Create a tensor, and set requires_grad=true to \
track computation with it")
x = torch.ones(2, 2, requires_grad=True)
print(x)

print("Perform a tensor operation: ")
y = x + 2
print(y)


print("y was created as a result of an operation, so it has a grad_fn")
print(y.grad_fn)

print("Do more operations on y")
z = y * y * 3
out = z.mean()
print(z, out)

print(".requires_grad_(...) changes an existing Tensor's requires_grad flag \
in-place. The input flag defults to False if not given.")

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


print("Gradients")
print("Because out contains a single scalar, out.backward() is equivalent to \
out.backward() is equivalent to out.backward(torch.tensor(1.)).")
out.backward()

print("Print gradients d(out)/dx")
print(x.grad)


