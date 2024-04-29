#This program goes through the multiple ways of creating a tensor with PyTorch and NumPy

import torch
import numpy as np

a = torch.FloatTensor(3,2) #Creates an uninitialized 3x2 tensor type Float
print(a)


a.zero_() #makes tensor all 0, inplace operation due to _ appended, actually modifies contents of "a"
print(a)

b = torch.FloatTensor([[1,2,3], [4,5,6]]) #Creates tensor with constructor using a python iterable
print(b)

n = np.zeros(shape=(3,2)) #Creates a 3x2 tensor with all zeros as the elements
print(n)

#.tensor() accepts from NumPy as an argument and creators tensor based on it's size
#creates DoubleTensor by default (64-bit float)
c = torch.tensor(n) 
print(c)

#To create a (32-bit float) to save performance since it's enough
z = np.zeros(shape=(3,2), dtype=np.float32)
print(torch.tensor(z))

#Equivalent but using PyTorch insetad of NumPy
x = np.zeros(shape=(3,2))
print(torch.tensor(x, dtype=torch.float32))
