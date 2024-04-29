import torch
import numpy as np

"""
BASIC OPERATIONS
Every function that has a trailing unnderscore(e,g, add_()) 
will do an inplace operation
"""

## creating a 2X3 2D tensor with random values assigned
x1 = torch.rand(2, 3) 
x2 = torch.rand(2, 3) 
print(x1)

## creating a 2X3 2D tensor with empty values assigned of dtype int
y = torch.empty(2, 3, dtype=torch.int) 
print(y)

## creating a 2X2X3 3D tensor with ones assigned of data type double
z = torch.ones(2, 2, 3, dtype = torch.double) 
print(z)

## default data type of a tensor is float32
print(x1.dtype)

## creating a PyTorch Tensor using a python list
m = torch.tensor([1, 2, 3, 4, 5], dtype=torch.double)
print(m) 

## demonstrating element wise addition of two PyTorch tensors
# k = x1 + x2
## OR
k = torch.add(x1, x2)
print("k =", k)

## demonstrating element wise subtraction of two PyTorch tensors
# k = x1 - x2
# OR
k = torch.sub(x1, x2)
print("k =", k)

## Inplace addition or subtraction can be done using the add_() or sub_() function
#  E.g. x1.add_(x2) will add all elements index wise of both tensors
#  and store it in x1

##  These similar element wise functions exist for
#  Multiplication: x1*x2; torch.mul(x1, x2); x1.mul_(x2)
#  Division: x1/x2; torch.div(x1, x2); x1.div_(x2)

print("\n\n")

"""
SLICING OPERATIONS 
Tensors provide slicing operations like numpy arrays
"""

t = torch.rand(5, 3)
print(t)
# printing all columns but all the rows
print(t[:, 0])
# printing all 2nd Row but all the colums
print(t[1, :])

## Printing item value of single tensor using the item() function
# the item() function can be used only if we have one elemtent in the tensor
# e.g. t[1, 1] 
print(t[1, 1].item())

print("\n\n")

"""
RESHAPING TENSORS
"""
## The view() function is used to reshape a tensor 
t1 = torch.rand(4, 4)
print("t1 =", t1)
# printing t2 which is a 1D vector with elements of t1(4X4)
t2 = t1.view(16)
print("t2 =", t2)

## If we do not know one of the dimensions to be set we can represent it by '-1' 
# and PyTorch will automatically decide that demension based on the other dimensions 
t3 = t1.view(-1, 8)
print("t3 =", t3)
## The size() function is used to find the size of a tensor 
print(t3.size())

print("\n\n")

"""
TENSORS AND NUMPY
Note: If the tensor is on the CPU and not the GPU then both objects 
(the numpy array and the tensor) will share the same memory location
and if one is modified even the other one will be.
"""

a = torch.ones(5)
print("a =", a)
## To convert a tensor to a numpy array the tensor.numpy() function is used
b = a.numpy()
print("b =", b)
## When we print the type() of b we can see it is a numpy.ndarray
print("The type of b is", type(b), "\n")

## Demonstrating the note by modifying the numpy array and tensor
print("Modifying a to see results in b as well")
a.add_(1)
print()
print("a =", a)
print("b =", b)

## To convert a numpy array to a tensor the torch.from_numpy() function is used
# default dtype is float32 of the tensor
# as an argument to the function
c = np.ones(5)
print("c =", c)
d = torch.from_numpy(c)
print("d =", d)

## When You need to optimize your variables in code 
# the requires_grad=True paramenter is used while creating a tensor