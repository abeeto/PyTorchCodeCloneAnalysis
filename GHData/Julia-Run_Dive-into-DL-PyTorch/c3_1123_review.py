import torch

# 一些创造tensor的基础函数#######################################
# x = torch.empty(3, 4)
# print(x, '\n', x.size(), x.shape)  # 一些极小值
#
# x = torch.zeros(3, 4)
# print(x, '\n', x.size(), x.shape)
#
# x = torch.ones(3, 4)
# print(x, '\n', x.size(), x.shape)
#
# x = torch.eye(3)
# print(x, '\n', x.size(), x.shape)
#
# x = torch.rand(3, 4)  # 0-1之间的随机数
# print(x, '\n', x.size(), x.shape)
#
# x = torch.randn(3, 4)  # 包含复数
# print(x, '\n', x.size(), x.shape)
#
# x = torch.arange(1, 10, 3)
# print(x, '\n', x.size(), x.shape)  # 和range用法一致
#
# x = torch.linspace(1, 10, 3)  # 等距，均匀分布，包尾
# print(x, '\n', x.size(), x.shape)
#
# x = torch.randperm(10)  # 打乱排序
# print(x, '\n', x.size(), x.shape)
#
# x = torch.tensor([1, 2, 3])  # np中的啊绕任意转换为tensor
# print(x, '\n', x.size(), x.shape)

# # tenor的加法#################################################
# a = torch.ones(3, 2, 4)
# b = torch.rand(3, 2, 4)
# y = a + b
# print(y, id(y))
# y = torch.add(a, b)
# print(y, id(y))
# a += b
# print(a, id(a))
# a.add_(b)
# print(a, id(a))
# r1 = torch.sum(y, dim=1)  # dim对应轴（0，1，2）。。倒着数就是（-1，-2，-3）
# # 针对某维度求和就是向该维度塌陷
# print(r1)
# r2 = torch.sum(y, dim=-2)
# print(r2)

# tensor的索引#########################################################
# a = torch.rand(2, 3, 4)
# print(a)
# x = a[1, :]
# print(x)
# x = a[:, 2, :]
# print(x)
# x = torch.index_select(a, 2, torch.tensor([1, 2]))  # dim=0,选前两个box
# print(x)
# x = torch.nonzero(a)
# print(x) # 返回索引
# # dim参数和加法的理解############################################################
# x = torch.sum(a, dim=2)  # - -2*3
# # dim=2,最内部的矩阵内的元素压缩求和，变成单一元素，去掉中括号
# print(x)
#
# x = torch.sum(a, dim=1)  # - -2*4
# # dim=1，最外面的每个box里，3*4沿着3向方向压缩成1*4，整体变成2*4
# print(x)
#
# x = torch.sum(a, dim=0)  # - -3*4
# # dim=0，最外面两个box叠加，对应元素求和
# print(x)
#
# # view # 广播
# a = torch.rand(2, 3, 4)
# print(a, id(a))
# x = a.view(-1, 12)  # 共享data，id不一致
# print(x, id(x))
# x = a.clone().view(-1, 12)  # 不共享data
# print(x, id(x))
# x = a + torch.tensor([1, 1, 1, 1])
# print(x)
# print(id(a))
# a.add_(x)
# print(a,id(a))
#
# numpy to tensor or reverse
import numpy as np

# a = torch.ones(5)
# b = a.numpy()
# print(type(a), type(b))
# print(a, b)
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(type(a), type(b))
# print(a, b)
# # 以上两种方案均share data
# a = torch.tensor([1, 2, 3]) # 不share data ，速度比较慢


# 自动求梯度

