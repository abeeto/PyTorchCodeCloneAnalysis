import torch
import numpy as np
# two ways to create a tensor (basically a matrix)
print("Creating a tensor")
x = torch.tensor([2, 1])
x = torch.rand(2,2)
y = torch.rand(2,2)
# y += x
print("adding tensors")
y.add_(x)
print(x)

# can also do slicing
print("Slciing tensors")
print(x[0,:])

# Converitng between numpy and tensor
a = torch.ones(5)
b = a.numpy()



