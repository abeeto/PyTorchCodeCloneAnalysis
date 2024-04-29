from __future__ import print_function
import torch
import numpy

#torch.eye(n,m=None,out=None)
#Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
#n (int) - Number of rows
#m (int, optional) - Number of columns. If None, defaults to n
#out (Tensor, optional) - Output tensor
x = torch.eye(3)
print(x)

#torch.from_numpy(ndarray) -> Tensor
#Creates a Tensor from a numpy.ndarray.
#The returned tensor and ndarray share the same memory.Modifications to the tensor will
#be reflected in the ndarray and vice versa.The returned tensor is not resizable.
a = numpy.array([1,2,3])
print (a)

t = torch.from_numpy(a)
print (t)
t[0] = -1
print (a) #a is not tensor, but ndarray


#torch.linspace(start, end, steps=100, out=None) -> Tensor
#Returns a one-dimensional Tensor of steps equally spaced points between start and end
#The output tensor is 1D of size steps
#start (float) - The starting value for the set of points
#end (float) - The ending value for the set of points
#steps (int) - Number of points to sample between start and end
#out (Tensor, optional) - The result Tensor
x = torch.linspace(3, 10, steps=5)
print (x)

x = torch.linspace(-10, 10, steps=5)
print (x)

x = torch.linspace(start=-10, end=10, steps=5)
print (x)


#torch.logspace(start, end, steps=100, out=None) -> Tensor
#Returns a one-dimensional Tensor of steps points logarithmically spaced between 10start and 10end
#The output is a 1D tensor of size steps
#start (float) - The starting value for the set of points
#end (float) - The ending value for the set of points
#steps (int) - Number of points to sample between start and end
#out (Tensor, optional) - The result Tensor
x = torch.logspace(start=-10, end=10, steps=5)
print (x)

x = torch.logspace(start=0.1, end=1.0, steps=5)
print (x)


#torch.ones(*sizes, out=None) -> Tensor
#Returns a Tensor filled with the scalar value 1, with the shape defined by the varargs sizes.
#sizes (int...) - a set of ints defining the shape of the output Tensor.
#out (Tensor, optional) - the result Tensor
x = torch.ones(2,3)
print (x)

x = torch.ones(5)
print (x)

x = torch.ones(1, 5)
print (x)


#torch.ones_like(input, out=None) -> Tensor
#Returns a Tensor filled with the scalar value 1, with the same size as input.
#input (Tensor) - The size of the input will determine the size of the output.
#out (Tensor, optional) - the result Tensor
input = torch.FloatTensor(2, 3)
print (input)
x = torch.ones_like(input)
print (x)


#torch.arange(start, end, step=1, out=None) -> Tensor
# torch.range(start, end, step=1, out=None) -> Tensor
#torch.zeros(*sizes, out=None) -> Tensor
#torch.zeros_like(input, out=None) -> Tensor
print (torch.zeros_like(input))