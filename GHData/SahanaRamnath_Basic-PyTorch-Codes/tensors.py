'''
------------------
TENSORS ON PYTORCH
------------------
'''

import torch
import numpy as np

# Creating tensors with different methods of initializations

# uninitialized
x = torch.empty(2,3)
print 'Uninitialized tensor : \n',x

# random initialization
x = torch.rand(2,3)
print '\nRandomly initialized tensor : \n',x

# initialized with zeros
x = torch.zeros(2,3,dtype=torch.long)
print '\nTensor initialized with zeros and type long : \n',x

# initialize from data
x = torch.tensor([[1,2,3],[4.5,6.9,7]]) # similar to numpy
print '\nTensor initialized with data : \n',x

# initialize with new_* method
x = torch.tensor((),dtype=torch.int32)
x = x.new_ones(2,3,dtype=torch.double) # can also do from existing not None tensor
print '\nTensor initialized with new_* method : \n',x

# initialize with existing tensor
x = torch.randn_like(x, dtype=torch.float) # can override datatype 
print '\nTensor initialized from existing tensor : \n',x

# get size of tensor
print '\nSize of tensor x : ',x.size()

# Operations
x = torch.ones(2,3,dtype=torch.int32)
y = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int32)
print '\nx : ',x,'\n y : ',y
print '\nx+y : ', torch.add(x,y) # or (x+y) or y.add_(x)
# similarly sub(), mul(), matmul() and div(){for dividing tensor by scalar value}

# Resize/Reshape
print 'x reshaped to 1D : ',x.view(-1)

# Numpy and PyTorch
x = torch.ones(5) # torch tensor
y = x.numpy() # numpy array
# the memory locations are shared and changing one will change the other
x.add_(1)
print x,y

# similarly, vice versa
a = np.ones(4)
b = torch.from_numpy(a)
a = a+1
print a,b
