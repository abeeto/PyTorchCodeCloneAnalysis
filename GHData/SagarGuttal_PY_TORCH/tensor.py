import torch
import numpy as np

# # empty tensors
# x = torch.empty(8)
# # print(x)

# # tensor with random 
# x = torch.rand(2,2)

# # zeros tensor
# x = torch.zeros(2,2)

# # ones tensors
# x = torch.ones(2,2)

# # Changing the dtype of a tensor
# x = x.dtype
# print(x) # --> torch.float32

# Y =  torch.ones(3,2, dtype=torch.int)
# print(Y.dtype)
# print(Y.size()) # Checking the size of a tensor

# # COnnstructing a tensor directly  from a list

# z = torch.tensor([2,5,3,6])
# print(z)

# # Tensor operations
# # Adding the tensor
# x = torch.rand(2,3)
# y = torch.rand(2,3)

# z1 = x + y
# z2 = torch.add(x,y)

# print(z1)
# print(z2)

# # adding the tensor elememts where inplace the elements
# y.add_(x) #elements of x will added to y its inplace operation
# print(y)

# #  Substracting the tennsor
# x1 = torch.rand(2,5)
# y1 = torch.rand(2,5)

# c1 = x1 - y1
# print(c1)
# c2 = torch.sub(x1,y1)
# print(c2)
# y1.sub_(x1)
# print(y1)

# # multiplication
# z1 = x * y
# z2 =  x.mul_(y)
# z3 = torch.mul(x,y)

# print(z1)
# print(z1)
# print(z1)

# # division
# z1 = x / y
# z2 =  x.div_(y)
# z3 = torch.div(x,y)

# print(z1)
# print(z1)
# print(z1)

# Torch slicing
x = torch.rand(5,3)
print(x)
print(x[:,0])# all rows and 0 th column
print(x[:,:2]) # all rows and 0 to 1 column
print(x[:2,:]) # first two rows and all columns

# selecting single element in tensor
print(x[1,1].item()) # note :- item function only work when single element is selelcted

# Reshaping the tensor
x = torch.rand(4,4)
y = x.view(16) # size ---> 1,16
z = x.view(8,2) # size ---> 8,2

# automatically detecting the right size by giving one value
y1 = x.view(-1,16) # size ---> 1,16
z1 = x.view(-1,8) # size ---> 2,8
print(x)
print(y1.size())
print(z1.size())




## Converting numpy to tensor 

a = torch.ones(5,3)
print(type(a))
b = a.numpy()
print(type(b))

# NOte :- tensor and numpy is both stores in same memorybite
# if one changes another also changes

print(a)
print(b)
a.add_(1)
print(a)


# converting tensor to numpy

x1 = np.ones(5)
x2 = torch.from_numpy(x1) 
# x3 = torch.from_numpy(x1, dtype=torch.float32) # we also  add  dtype 

print(x1.dtype)

print(x2.dtype)


## How to move tensor to GPU for  computing

#Checking for gpu available
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # passing this tensor to GPU(cuda)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)

    #again converting tensor to numpy we havr to change the device for numpy
    c = z.to("cpu")
    m=c.numpy()
    print(type(m))