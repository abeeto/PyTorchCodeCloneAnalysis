# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#getting-started
from __future__ import print_function
import torch

#Construct an empty matrix of 5x3
x = torch.empty(5, 3)
print(x)

#Construct a matrix of 5x3 with random data
X = torch.rand(5, 3)
print(x)

#Construct a matrix of 5x3 with zero's
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)

#Get Size
print(x.size())

#Add up
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y)) #same as above

result = torch.empty(5, 3)
torch.add(x, y, out=result) #Copy to an a new variable
print(result)

#In place
y.add_(x)
print(y)

# NumPy-like indexing 🐔
print(x[:, 1])

#Resize
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#Single item
x = torch.randn(1)
print(x)
print(x.item())

#NumPy bridge
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!