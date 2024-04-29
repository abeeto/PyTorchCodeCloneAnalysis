import torch
import numpy as np

a = torch.randn(2, 3)

#################
print(a.type())

###################
print(isinstance(a, torch.FloatTensor))


######################
#a = a.cuda() #搬运到GPU上

#Create Dimension 0 Data
data = torch.tensor(1.0)

#shape
print(data.shape)

#size
print(a.size())

#Create Dimension 1 Data
data = torch.tensor([1.1])
print(data)

data = torch.tensor([1.1, 2.2])
print(data)

data = torch.FloatTensor(1) #生成一个随机的FloatTensor数据
print(data)

data = torch.FloatTensor(2)
print(data)

data = np.ones(2)
print(data)

data = torch.from_numpy(data)
print(data)

#Create Dimension 2 Data
a = torch.randn(2, 3)
print(a)

print(a.shape)

print(a.size(0))

print(a.size(1))

print(a.shape[1])

#Create Dimension 3 Data
a = torch.randn(1, 2, 3)

print(a.shape)

print(a[0])

print(list(a.shape))

#Create Dimension 4 Data
a = torch.randn(2, 1, 28, 28)

print(a.shape)

#获得数据的大小
print(a.numel)

#获得数据维度
print(a.dim())

