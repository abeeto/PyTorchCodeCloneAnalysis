import torch
import numpy as np

'''
 part1 
 https://www.bilibili.com/video/av15997678/?p=7&t=487
'''
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    'numpy\n',np_data,
    '\ntorch\n',torch_data,
    '\ntensor2array\n',tensor2array,
)

data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)   #32bit
# http://pytorch.org/docs/torch.html#math-operations
# the website shows some functions in pytorch which does many mathematic operations.


print(
    'abs\n',
    '\nnumpy:\n',np.abs(data),
    '\ntorch:\n',torch.abs(tensor)
)

print(
    'sin\n',
    '\nnumpy:\n',np.sin(data),
    '\ntorch:\n',torch.sin(tensor)
)

print(#the averange of several numbers
    'mean\n',
    '\nnumpy:\n',np.mean(data),
    '\ntorch:\n',torch.mean(tensor)
)

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data = np.array(data)
print(
    'numpy\n',np.matmul(data, data),
    '\nnumpy\n',data.dot(data),
    '\ntorch\n',torch.mm(tensor,tensor),
#    '\ntorch\n',tensor.dot(tensor),#in the new pytorch version,this function expects 1D tensors
)



'''
 part2
 https://www.cnblogs.com/wj-1314/p/9830950.html
'''

# The data structure of Tensor
# torch.FloatTensor
print('\ntorch.FloatTensor')
a = torch.FloatTensor(2,3)
b = torch.FloatTensor([2,3,4,5])   #tensor([2., 3., 4., 5.]),the point follows the number shows that the type of the number is float
print (a,a.size())
print (b,b.size())

# torch.IntTensor
print('\ntorch.IntTensor')
a = torch.IntTensor(2,3)
b = torch.IntTensor([2,3,4,5])
print (a,a.size())
print (b,b.size())

# torch.randn
#Generate a ranfom floating point number that satisfies a normal distribution with a mean of 0 and a variance of 1
print('\ntorch.randn')
a = torch.randn(2,3)
print (a,a.size())

# torch.range
print('\ntorch.range')
a = torch.range(2,8,1)
print (a,a.size())

# torch.zeros
print('\ntorch.zeros')
a = torch.zeros(2,3)
print (a,a.size())


# Tensor's operation
# torch.abs
print('\ntorch.abs')
a = torch.randn(2,3)
b = torch.abs(a)
print (a,a.size())
print (b,b.size())

# torch.add
print('\ntorch.add')
a = torch.randn(2,3)
b = torch.randn(2,3)
c = torch.add(a,b)
d = torch.add(c,10)
print (a,a.size())
print (b,b.size())
print (c,c.size())
print (d,d.size())

# torch.clamp
print('\ntorch.clamp')
a = torch.randn(2,3)
b = torch.clamp(a,-0.1,0.1)
print (a,a.size())
print (b,b.size())

# torch.div
print('\ntorch.div')
a = torch.randn(2,3)
b = torch.randn(2,3)
c = torch.div(a,b)
d = torch.div(c,10)
print (a,a.size())
print (b,b.size())
print (c,c.size())
print (d,d.size())

# torch.mul
print('\ntorch.mul')
a = torch.randn(2,3)
b = torch.randn(2,3)
c = torch.mul(a,b)
d = torch.mul(c,10)
print (a,a.size())
print (b,b.size())
print (c,c.size())
print (d,d.size())

# torch.pow
print('\ntorch.pow')
a = torch.randn(2,3)
b = torch.pow(a,2)
print (a,a.size())
print (b,b.size())

# torch.mm
print('\ntorch.mm')
a = torch.randn(2,3)
b = torch.randn(3,2)
c = torch.mm(a,b)
print (a,a.size())
print (b,b.size())
print (c,c.size())

# torch.mv
print('\ntorch.mv')
a = torch.randn(2,3)
b = torch.randn(3)
c = torch.mv(a,b)
print (a,a.size())
print (b,b.size())
print (c,c.size())



