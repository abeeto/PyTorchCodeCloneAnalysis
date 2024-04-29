from __future__ import print_function
import torch
#costruisce un tensore con valori random, 5 righe e 3 colonne
#la struttura tensore è un array numpy supportato in questa libreria per essere gestito su GPU
#vale la pena se tutte le operazioni da farci sono su GPU

#5x3 initialized matrix
x = torch.empty(5,3)
print(x)

#random initialized matrix
x = torch.rand(5,3)
print(x)

#matrix filled zeros and dtype long
x =  torch.zeros(5,3,dtype = torch.long)
print(x)

#construct a tensor from data
x = torch.tensor([5.5,3])
print(x)

#create tensor of ones
x = x.new_ones(5,3,dtype=torch.double)
print(x)

#create tensor from an existing tensor
x = torch.randn_like(x,dtype=torch.float)
print(x)

#Operations
print("OPERATIONS")

y = torch.rand(5,3)
#ADD
print("ADD: ")
#1)
print(x+y)

#2)
print(torch.add(x,y))

#3)providing an output tensor as argument
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#4)
y.add_(x)
print(y)


#RESIZE
print("RESIZE: ")
#view is an operation that is used to change the point of view of a tensor.
#if i say: x.view(15) that means that i want to see the tensor as a array with dimension 15

y = x.view(15)
z = x.view(-1,5)
print(x.size(),y.size(),z.size())


#one element sensor print

z = torch.randn(1)
print(z)
print(z.item())


#NUMPY BRIDGE - CONVERTING TENSOR->NUMPY AND NUMPY->TENSOR
print("CONVERT tensor->numpy numpy->tensor")
a = torch.ones(5)
b = a.numpy()
print(a)
print(b)
#when i modify ones, the same change is applied on the other one
a.add_(1)
print(a)
print(b)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#if it is not decleared, the tensor is run on the cpu (the standard device)
#it is possible to move the tensor in another device by the .to method

if torch.cuda.is_available():
    device = torch.device("cuda") # a CUDA device object
    y = torch.ones_like(x, device=device)# directly create a tensor on GPU
    x = x.to(device)# or just use strings ``.to("cuda")``
    z = x+y
    print(z)
    print(z.to("cpu",torch.double))