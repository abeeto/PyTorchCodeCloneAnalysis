# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch

#------------------Tensors--------------------

#5x3 matrix uninitialized
x = torch.empty(5,3)
print(x)

#Randomly initialized matrix
y = torch.rand(5,3)
print(y)

#Matrix filled with zeros and dtype long
z = torch.zeros(5,3,dtype=torch.long)
print(z)

#Constructing the tensor from data
a = torch.tensor([5.5,3])
print(a)

#Checking the size of tensor
print(x.size())


#-----------------Operations-------------------

#Addition operation
print(y + x)

print(torch.add(x,y))

#Providing output tensor as argument
result = torch.empty(5,3)
torch.add(x,y,out=result)
print("Result of x and y\n",result)

#addition in place
y.add_(x)
print("Inplace addition \n", y)


#Numpy like indexing
#prints all rows of column 1
print(x[:,1])


#resize/ reshape tensor
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(),y.size(),z.size())


#For 1 element tensor, this gives python number
x = torch.randn(1)
print(x)
print(x.item())























