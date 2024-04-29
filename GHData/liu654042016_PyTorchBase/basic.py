import torch
import numpy as np


#test
#创建一个未初始化矩阵
empty = torch.empty(5,3)
#创建一个空矩阵
zeros = torch.zeros(5,4)
#创建指定范围整型矩阵
randInt = torch.randint(1, 10, (5, 4))
#创建随机矩阵
rand = torch.rand(5, 4)
#转为tensor类型
x = torch.tensor([5, 5, 3])
#转为1矩阵
x = x.new_ones((5, 3))
#生成形状相同， 均值为0，方差为1的标准正态分布数据
x = torch.rand_like(x, dtype=torch.float)
print(x)
#查看维度
print(x.size())
#同numpy的reshape
print(x.view(15))
#从torch 转为Numpy，内存共享，改变一个，另一个同时改变
a = torch.ones(5, 3)
b = a.numpy()
b[0][0] = 100
print(a)
#从numpy 转为torch内存共享，改变一个另一个同时改变
a = np.ones((5, 3))
b = torch.from_numpy(a)
b[0][0] = 100
print(a)
#查看gpu版是否安装
print(torch.cuda.is_available())
#扩展维度
#返回一个新的张量， 对输入既定位置插入位置为1
#torch.unsqueeeze
x = torch.Tensor([1,2,3,4])
print(x)
print(x.size())
print(x.dim())
print(x.numpy())



