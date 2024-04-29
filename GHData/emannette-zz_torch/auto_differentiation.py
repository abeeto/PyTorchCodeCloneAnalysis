import torch;

# create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True);
print(x);

# do an operation of tensor
y = x + 2;
print(y);

# y was created as a result of an opertaion so it has a grad_fn
print(y.grad_fn);

# perform some more operations on y
z = y * y * 3
out = z.mean();
print(z, out);

'''
.requires_grad_(...) changes an existing tensor's requires_grad
flag in-place. The input flag defaults to False if not given
'''
a = torch.randn(2, 2);
a = ((a *3) / (a - 1));
print(a.requires_grad);
a.requires_grad_(True);
print(a.requires_grad);
b = (a * a).sum();
print(b.grad_fn);

'''
GRADIENTS
'''
print("GRADIENTS");
out.backward(); # This is the equivalent to out.backward(torch.tensor(1))
# print gradients d(out)/dx
print(x.grad);

# an example of Jacobian-vector product
x = torch.randn(3, requires_grad=True);
y = x * 2;
while y.data.norm() < 1000:
    y = y * 2;
print(y);

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float);
y.backward(v);
print(x.grad);

print(x.requires_grad);
print((x ** 2).requires_grad);
with torch.no_grad():
    print((x ** 2).requires_grad);
