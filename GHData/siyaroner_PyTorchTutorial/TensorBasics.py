import torch
import numpy as np

# #1D empty tensor
# x=torch.empty(3)
# #2D empty tensor
# x=torch.empty(2,3)
# #3D empty tensor
# x=torch.empty(2,3,1)
# #random tensor values
# x=torch.rand(2,3)
# #Zeros tensor
# x=torch.zeros(2,3)
# #Ones tensor
# x=torch.ones(2,3)
# #Ones tensor and specify data type
# x=torch.ones(2,3,dtype=torch.float32)
# print(x.dtype)
# print(x.size())
# x=torch.tensor([3,4.3])
# print(x)
# x=torch.rand(3,4)
# y=torch.rand(3,4)
# print(x[1,:])  #slicing
# print(x[1,3])
# print(x.view(2,6)) #reshaping
# print(x.view(-1,3)) #if we don't want to specify coloumn/row number then we should simply type -1 for it
# print(x.view(6,-1))
# z=x+y
# z=x-y
# z=x*y
# z=x/y
# z=torch.add(x,y)
# z=torch.sub(x,y)
# z=torch.mul(x,y)
# z=torch.div(x,y)
# print(z)
# y.add_(x) # _ operation is inplace operation so it will change y
# print(y)

# a=torch.ones(5)
# print(a)
# b=a.numpy()
# print(type(b))
# a+=1  #this will change b also because they are share same location on cpu
# print(a)
# print(b)

x=torch.ones(3, requires_grad=True) # requires_grad by default is false. And will tell pytorch that it will need to 
                                    # calculate the gradients for this tensor later in your optimization steps
