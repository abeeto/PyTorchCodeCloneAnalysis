# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:56:17 2020

@author: LUI8WX
"""
###############################################################################################
#                                       Basic operations
###############################################################################################


from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)  

print(x.size())

y = torch.rand(5, 3)

print(y)

print(x + y)


# y = torch.rand(5, 2)
# print(x + y)
# RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# in-place add
y.add_(x)
print(y)

# indexing like numpy
print(x)
print(x[:, 1])


# view for resize or reshape
x = torch.randn(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(z)
print(x.size(), y.size(), z.size())


# extract numbers for scalar tensor
x = torch.randn(1)
print(x)
print(x.item())

x = torch.randn(2)
print(x)
print(x.item())
# ValueError: only one element tensors can be converted to Python scalars




###############################################################################################
#                                       Conversion with numpy
###############################################################################################


# convert to numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)

print(a)

# b changes as a changes
print(b)



# convert to tensors
import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)
print(a)
# b changes as a changes
print(b)


###################
#  convert <-> same mem location
###################



###############################################################################################
#                                       CUDA
###############################################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print(device)
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

device = torch.device("cuda")          # a CUDA device object
device
y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
y

x = x.to(device) 
x

z = x + y

z = z.to("cpu")
z
z = z.to("cpu", torch.double)
z
