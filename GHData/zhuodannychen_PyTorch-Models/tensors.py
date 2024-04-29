#python
#pytorch tensors
#Tensor is just a matrix, or a numpy array, but better : )

from __future__ import print_function
import torch
import numpy as np

#creates tensor with size 5,3
x = torch.Tensor(5,3)
#other types
y = torch.ones(5,3), # or torch.zeros
z = torch.empty(5,3) # or torch.rand

a = torch.tensor([4,6]) # specific data


#convert tensor to numpy
x = torch.Tensor(5,3)
y = torch.numpy(x)
#convert numpy to tensor
x = np.array([1,2,3])
y = torch.from_numpy(x)

#basic operation
x = torch.Tensor(5,3)
y = torch.rand(5,3)
print(x + y)

x = torch.Tensor(5,3)
y = torch.rand(5,3)
# same as x = x + y
print(x.add_(y))

#extra specifics
x = torch.Tensor(5,3, dtype=torch.float)
#float value
