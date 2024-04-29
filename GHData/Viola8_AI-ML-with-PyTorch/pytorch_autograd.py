import torch

a = torch.ones(2, 2, requires_grad=True) # it starts to track all operations on it.
print(a)
b = a + 5   # tensor([[6., 6.],
print(b)    #        [6., 6.]], grad_fn=<AddBackward0>)
# Each tensor has a .grad_fn attribute that references a Function that has created the Tensor.
print(b.grad_fn) # <AddBackward0 object at 0x00001C058CB79D0>; 'b' was created as a result of an operation, so it has a grad_fn
c = b * b * 2
result = c.mean() # tensor([[72., 72.],
print(c, result)  # [72.,72.]], grad_fn=<MulBackward0>) tensor(72., grad_fn=<MeanBackward0>)

a = torch.randn(3, 3)
a = ((a * 4) / (a - 2)) # .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
print(a.requires_grad)   # False since the input flag defaults to False if not given.
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn) # <SumBackward0 object at 0x0000279B20776D0>

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y) # tensor([ - 766.6851, 905.2292, 363.6300], grad_fn=<MulBackward0>)
# torch.autograd could not compute the full Jacobian directly, but ...
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v) # pass the vector to backward as argument if we just want the vector-Jacobian product
print(x.grad) # Output: tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])

# To prevent tracking history (and using memory), we can also wrap the code block in with torch.no_grad():
print(x.requires_grad)  # True
print((x ** 2).requires_grad)  # True
with torch.no_grad():
    print((x ** 2).requires_grad)  # False

# Call .detach() to detach a tensor from the computation history, and to prevent future computation from being tracked:
print(x.requires_grad)   # True
y = x.detach()
print(y.requires_grad)  # False
print(x.eq(y).all())    # tensor(True)
