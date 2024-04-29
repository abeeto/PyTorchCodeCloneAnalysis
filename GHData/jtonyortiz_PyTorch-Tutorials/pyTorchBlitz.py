from __future__ import print_function
import torch

#Print an uninitialized 5x3 matrix:
x = torch.empty(5,3)
print(x)


#Construct a randomly initialized matrix:
x = torch.rand(5,3)
print(x)


#Consturct a matric filled with zeros and of dtype long:
x = torch.zeros(5,3, dtype=torch.long)

#Construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

#Create a tensor based on an existing tensor. These methods will resule properties of the input tensor, e.g. dtype, unless the values are provided by the user:

x = x.new_ones(5, 3, dtype=torch.double) #new* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float) #override dtype!
                                           #result has the same size
#Obtain the size
print(x.size())

#operations:

#Addition: Syntax 1
y = torch.rand(5,3)
print(x + y)

#Addition Syntax 2:
print(torch.add(x, y))

#Provide an output tensor as an argument:
result  = torch.empty(5,3)
torch.add(x, y, out=result)
print(result)

#Standard NumPy indexing:
print(x)
print(x[:, 1])


