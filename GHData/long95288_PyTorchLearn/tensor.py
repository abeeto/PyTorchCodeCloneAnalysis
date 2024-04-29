# coding=utf-8
import torch
from torch.autograd import Variable

# 向量
a = torch.FloatTensor([1])
b = torch.FloatTensor([2])
print(a+b)

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5,3,dtype=torch.long)
print(x)

# 创建相同形状的
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 获得维度信息
print(x.size())

x = torch.randn(2,1,7,3)
print(x)
conv = torch.nn.conv2d(1,8,(2,3))
res = conv(x)
print(res.shape)


