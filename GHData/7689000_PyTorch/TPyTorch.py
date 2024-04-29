import torch
import torchvision
import numpy as np

x = [12,23,34,45,56,67,78]
print(torch.is_tensor(x))
print(torch.is_storage(x))

y = torch.randn(1,2,3,4,5)
print(torch.is_tensor(y))
print(torch.is_storage(y))
print(torch.numel(y)) # the total number of elements in the input Tensor

print(torch.zeros(4,4))
print(torch.numel(torch.zeros(4,4)))

print(torch.eye(3,4))
print(torch.eye(5,4))

x1 = np.array(x)
print(x1)
print(torch.from_numpy(x1))

torch.linspace(2,10,steps=25) # linear spacing
torch.logspace(start=-10,end=10,steps=15) # logarithmic spacing

print(torch.rand(10)) # 10 uniformly-distributed random values ranging between 0 and 1
print(torch.rand(4,5)) # 4x5 uniformly-distributed random values ranging between 0 and 1
print(torch.randn(10)) # 10 normally-distributed random values with mean=0 and standard deviation=1
print(torch.randn(4,5)) # 4x5 normally-distributed random values with mean=0 and standard deviation=1

print(torch.arange(10,40)) # default step size = 1
print(torch.arange(10,40,2)) # step size = 2

torch.randperm(10) # randomly permute from 0 to 10.

d = torch.randn(4,5)
print(d)
print(torch.argmin(d,dim=1))
print(torch.argmax(d,dim=1))

print(torch.zeros(4,5)) # create a 2D tensor filled with zero values.
print(torch.zeros(10)) # create a 1D tensor filled with zero values.

x = torch.randn(4,5)
a = torch.cat((x,x)) # by default, dim=0, along rows
b = torch.cat((x,x),dim=1) # along columns
print(a)
print(a.shape)
print(b)
print(b.shape)

a = torch.cat((x,x,x),0) # concatenate x n times over row
b = torch.cat((x,x,x),1) # concatenate x n times over column
print(a)
print(a.shape)
print(b)
print(b.shape)

a = torch.randn(4,4)
print(a)
print(torch.chunk(a,2))
print(torch.chunk(a,2,0))
print(torch.chunk(a,2,1))

b = torch.Tensor([[11,12],[23,24]])
print(b)
print(torch.gather(b, 1, torch.LongTensor([[0,0],[1,0]])))

a = torch.randn(4,4)
print(a)
indices = torch.LongTensor([0,2])
print(indices)
print(torch.index_select(a,0,indices))
print(torch.index_select(a,1,indices))

# identify null input tensors using nonzero function
torch.nonzero(torch.tensor([10,0,23,0,0.0]))

# splitting the tensor into two small chunks.
print(torch.split(torch.tensor([12,21,34,32,4,54,56,65]),2))
# splitting the tensor into three small chunks.
print(torch.split(torch.tensor([12,21,34,32,4,54,56,65]),3))

x = torch.randn(4,5)
print(x)
print(x.t()) # transpose
print(x.transpose(1,0)) # transpose

x = torch.randn(4,5)
print(x)
print(torch.unbind(x,1)) # Remove the column dimension
print(torch.unbind(x)) # Remove the row dimension

x = torch.randn(4,5)
print(x)
print(torch.add(x,20)) # scalar addition
print(torch.mul(x,2)) # scalar multiplication
z = torch.randn(2,2)
print(z)
beta = 0.7456
intercept = torch.randn(1)
print(intercept)
y = torch.add(intercept, torch.mul(z,beta)) # y = intercept + (beta * z)
print(y)

torch.manual_seed(1234)
print(torch.randn(5,5))
torch.manual_seed(1234)
print(torch.ceil(torch.randn(5,5)))
torch.manual_seed(1234)
print(torch.floor(torch.randn(5,5)))

torch.manual_seed(1234)
print(torch.clamp(torch.floor(torch.randn(5,5)), min=-0.3, max=0.4))
torch.manual_seed(1234)
print(torch.clamp(torch.floor(torch.randn(5,5)), min=-0.3))
torch.manual_seed(1234)
print(torch.clamp(torch.floor(torch.randn(5,5)), max=0.4))

torch.manual_seed(1234)
a = torch.randn(5,5)
print(a)
print(torch.exp(a))
torch.manual_seed(1234)
a = torch.randn(5,5)
print(a)
print(torch.log(a))
torch.manual_seed(1234)
a = torch.randn(5,5)
print(a)
print(torch.pow(a,2))

torch.manual_seed(1234)
x = torch.randn(5,5)
print(x)
print(torch.frac(torch.add(x,10)))

torch.manual_seed(1234)
a = torch.randn(5,5)
print(a)
print(torch.sigmoid(a))
torch.manual_seed(1234)
a = torch.randn(5,5)
print(a)
print(torch.sqrt(a)) # finding the square root of the values

