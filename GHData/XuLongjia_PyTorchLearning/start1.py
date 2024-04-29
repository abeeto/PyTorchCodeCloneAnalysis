
#https://www.bilibili.com/video/av15997678
#本节课学习一下numpy与torch处理数据的不同


import torch

import numpy as np 

np_data = np.arange(9).reshape((3,3))
torch_data = torch.from_numpy(np_data) #将np的数据转换成torch的数据
tensor2array = torch_data.numpy() #再转换回来

print(np_data)
print(torch_data)
print(tensor2array)


#abs
data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)  #32bit
print(tensor)
print(torch.abs(tensor))
print(torch.sin(tensor))
print(torch.mean(tensor))

#numpy的矩阵运算：
print(np_data.dot(np_data))

#toroch的矩阵运算：
print(torch_data.mm(torch_data))


