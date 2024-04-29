import torch
import numpy as np

# 1. INTRODUCTION OF TENSORS
x = torch.empty(2, 3)
z = torch.rand(2, 2)
x1 = torch.ones(2, 2, dtype=torch.int)
x2 = torch.ones(2, 2, dtype=torch.float16)

# print(x1.dtype)
# print(x1.size())

# can construct tensor from data (e.g. from a python list)
y = torch.tensor([2.5, 0.1])
# print(y)

# create random tensor of size 2 x 2
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
# addition
z = x + y
print(z)
# or
z = torch.add(x, y)
print(z)

# inplace manipulation of tensors (underscore represents inplace manipulation)
y.add_(x)
print(y)

# multiplication
c = x * y
c = torch.mul(x, y)
y.mul_(x)

# elementwise division
b = x / y
b = torch.div(x, y)
y.mul_(x)

# slicing operations (just like numpys)
a = torch.rand(5, 3)
print(a)
print(a[:, 0])
# select a single element from a tensor
print(a[1, 1].item())

# reshaping a tensor
q = torch.rand(6, 4)
print(q)
t = q.view(4, 6)
print(t)
# use -1 to let pytorch automatically determine the number of rows or columns
t = q.view(-1, 8)
print(f't:{t}')
print(t.size())
# or use -1 to convert tensor to a one-dimensional tensor
s = q.view(-1)
print(s)

# convert torch tensor to numpy array
m = torch.ones(5)
print(m)
n = m.numpy()
print(type(n))
# we have to be careful because if the tensor is on the CPU not the GPU, both objects will share the same memory
# location so if we modify one, we will be modifying the other also
m.add_(1)
print(m)
print(n)

# convert numpy array to torch tensor (by default it converts to torch floats)
o = np.ones(5)
p = torch.from_numpy(o)
print(p)
# we have to be careful because if the tensor is on the CPU not the GPU, both objects will share the same memory
# location so if we modify one, we will be modifying the other also
o += 1
print(o)
print(b)

# 2. WE CAN RUN THESE OPERATIONS ON THE GPU IF WE HAVE CUDA AVAILABLE
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    # or
    x = torch.ones(5)
    y = t.to(device)
    # this operation will then be executed on the GPU
    z = x + y
    # the following code will return an error because numpy can only handle CPU tensors
    # z.numpy()
    # to move the operation back to the CPU
    z = z.to("cpu")

# specifying requires_grad = True tells Pytorch that it will need to calculate the gradient for this tensor
# later in the optimization steps
x = torch.ones(5, requires_grad=True)
print(x)
