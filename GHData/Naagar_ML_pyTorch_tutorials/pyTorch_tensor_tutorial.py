# pyTorch_tensor_tutorial

import torch 
# ===================== #
# ===================== #
# x = torch.zeros([2, 3, 4])

# x = torch.ones([2, 3, 4])
# y = torch.ones([50,60 ])
# # print(x)
# # print(x.shape, x.size())
# print(y.shape, y.size())
# print(y)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
my_tensor = torch.tensor([[2, 3, 4], [4, 5, 6]], dtype=float, device='cpu', requires_grad=True)
# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.size())
# print(my_tensor.requires_grad)
# print(my_torch.layout) # how the data is stored in memory 



# other common initilazation 
# x = torch.empty(size=(3, 4))  # values will be random
# # print(x)
# x = torch.zeros(3, 3)
# # print(x)

# x = torch.rand((3,3))
# # print(x)
# x = torch.eye(5, 5)  # Identity matrix 
# print(x)

# x = torch.arange(start=0, end=5, step=1)
# print(x)

x = torch.linspace(start=0.1, end=1, steps=10 )
print(x)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)

x = torch.diag(torch.ones(3))
print(x)

# How to initialize and convert tensors to other types(int, float, double, boolean)

torch1 = torch.arange(4)

print(torch1.bool())
print(torch1.short())
print(torch1.double())
print(torch1.half())
print(torch1.float())
print(torch1.long())

import numpy as np

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 6])


# addition

z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y 

z = x - y 

z = torch.true_divide(x, y)


t = torch.zeros(3)
t.add_(x)

t += x 

z = x.pow(2)
z = x ** 2

# Simple 

z = x > 0
z = x < 0

# Matrix Multiplication 

x1 = torch.rand(2, 5)
x2 = torch.rand(5, 3)
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# matrix exponention 
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))


z = x * y
print(z)

z = torch.dot(x, y)
print(z)

# batch Matrix Multiplication 

batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand(batch, n, m)
tensor2 = torch.rnad(batch, m, p)
out_bmm = torch.bmm(tnesor1, tensor2)



# Example of broadcasting 

x1 = torch.rand(5,5)
x2 = torch.rand(1, 5)

z = x1 - x2

z = x1 ** x2

# other useful tensor operations 

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)

values, indices = torch.min(x, dim=0)
abs_x= torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)

mean_x = torch.mean(x.float(), dim=0)

z = torch.eq(x, y)

sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)


#  batch Indexing
batch_size = 10 
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape) 

print(x[:, 0])

print(x[2, 0:10]) # will creat a list 0-9
x[0, 0] =100 # assigne

# fancy indexing 

x = torch.arange(10)
indices = [2, 3, 4]
print(x[indices])

x = torch.rand(3, 5)
rows = torch.rand([1, 0])
columns = torch.rand([4, 0])

print(x[rows, columns].shape)


x = torch.arange(10)

print(x[(x<2) & (x>8)])

print(x[x.remainder(2) == 0])

print(torch.where(x>5, x, x*2)) #   if  x>5  else x*2

print(torch.tensor([0, 2, 4, 5, 6]).unique())

print(x.ndimension())  ##  5x5x5 
print(x.numal())

## Reshape a Tensor 

x = torch.arange(9)

x_3x3 = v.view(3,3) ## act on continous vector, memory block needs continous memoery block 

x_3x3 = x.reshape(3,3)

y = x_3x3.t() ## [0, 3, 6, 1, 7, 2, 5, 8, 9]
print(y)

print(y.view(9)) # error view size 
print(y.contigous().view(9))


x1 = torch.rand(2, 5)
x2 =torch.rnad(2, 5)
print(torch.cat(x1, x2, dim=0).shape())

z = x1.view(-1) # just want to flatten full tensor


batch = 64

x = torch.rand(batch, 2, 5)

z = x.view(batch, -1)
print(z,shape)

z = x.permute(0, 2, 1) # permutation of the tensor in dimensions

print(z.shape)

x = torch.arange(10)

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)

print(z.shape)


print(torch.empty(0))









