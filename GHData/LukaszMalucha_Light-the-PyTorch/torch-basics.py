# -*- coding: utf-8 -*-


import torch
import numpy as np


arr = ([[1,2],[3,4]])


## Convert to tensor

tensor = torch.Tensor(arr)


## Same as np 
    
ones = np.ones((2,2))

ones = torch.ones((2,2))


torch.rand(2,2)


################################################################## Seed with CPU

np.random.seed(0)
np.random.rand(2,2)

torch.manual_seed(0)
torch.rand(2,2)


####################################################### Settign up seed with GPU

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
    
    
################################################### Numpy array into torch array
np_array = np.ones((2,2))

torch_tensor = torch.from_numpy(np_array)

## Can't convert int8

np_array_new = np.ones((2,2), dtype=np.int8)

torch_tensor = torch.from_numpy(np_array_new)

## int64    LongTensor
## int32    IntTensor
## uint8    ByteTensor
## float64  DoubleTensor
## double   DoubleTensor
## float32  FloatTensor

########################################################## Torch tensor to numpy

torch_tensor = torch.ones(2, 2)

torch_to_numpy = torch_tensor.numpy()


############################################################# CPU / GPU Toggling

tensor_cpu = torch.ones(2,2)

if torch.cuda.is_available():
    tensor_cpu.cuda()
    
tensor_cpu.cpu() 


############################################################## Matrix Operations

c = torch.add(a,b)  

a.sub(b)
a.sub_(b)       ## change exisitng tensor with underscore

torch.mul_(a,b)
b.div_(a)
torch.div(b,a)

a.mean(dim=0) ## pass the dim





###################################################################### Variables

from torch.autograd import Variable

a = Variable(torch.ones(2,2), requires_grad=True)



###################################################################### Gradients

x = Variable(torch.ones(2), requires_grad=True)

y = 5 * (x + 1) ** 2

x.grad








    