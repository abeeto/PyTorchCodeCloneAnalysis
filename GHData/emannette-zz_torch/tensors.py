from __future__ import print_function
import torch
import numpy as np

# construct a 5x3 matrix, uninitialized
x = torch.empty(5, 3);
print(x);

# construct a randomly initialized matrix
x = torch.rand(5, 3);
print(x);

# construct a matrix filled zeros and of data type long
x = torch.zeros(5, 3, dtype=torch.long);
print(x);

# construct a tensor directly from data
x = torch.tensor([5.5, 3]);
print(x);

'''
create a tensor based on an existing tensor.
these methods will reuse properties of the input tensor (e.g. dtype)
unless new values are provided by the user
'''
x = x.new_ones(5, 3, dtype=torch.double);
print(x);

# override the previous dtype
x = torch.randn_like(x, dtype=torch.float);
print(x);

# get the size
# torch.size is a tuple, supports all tuple operations
print(x.size());

# addition Syntax
y = torch.rand(5, 3);
print(x + y);
print(torch.add(x, y));

# providing an output tensor as an argument
result = torch.empty(5, 3);
torch.add(x, y, out=result);
print(result);

# adds x to y
y.add_(x);
print(y);

'''
any operation that mutates a tensor in-place is post-fixed with an _
for example: x.copy_(y), x.t_(), will change x
'''

# can use standard NumPy-like indexing with all bells and whistles
print(x[:, 1]);

# to resize/reshape tensor, can use torch.view
x = torch.randn(4, 4);
y = x.view(16);
z = x.view(-1, 8)
print(x.size(), y.size(), z.size());

# use .item() to get the value of a single element tensor
x = torch.randn(1);
print(x);
print(x.item());

# convert a torch tensor to a NumPy array
a = torch.ones(5);
print(a);

b = a.numpy();
print(b);

# see how the numpy array changed in value
a.add_(1);
print(a);
print(b);

# convert a NumPy array to a torch tensor
a = np.ones(5);
b = torch.from_numpy(a);
np.add(a, 1, out=a);
print(a);
print(b);

# tensors can be moved onto any device using the .to method
if torch.cuda.is_available():
    device = torch.device("cuda");         # a CUDA device object
    y = torch.ones_like(x, device=device); # directly create a tensor on GPU
    x = x.to(device);                      # or just use strings ''.to("cuda")''
    z = x + y;
    print(z);
    print(z.to("cpu", torch.double));      # ''.to'' can also change dtype together
