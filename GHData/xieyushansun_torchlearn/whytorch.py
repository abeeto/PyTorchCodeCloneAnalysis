import torch

# 形成一个tensor
# x = torch.tensor([2.1, 2])
# 随机生成矩阵
# y = torch.rand(3, 4)
# print(y)

# 两种加法，注意第二种赋值方式，
# x = torch.rand(3, 4)
# y = torch.rand(3, 4)
# result = torch.add(x, y)
# result_temp = torch.empty(3, 4)
# torch.add(x, y, out=result_temp)
# print(result)
# print(result_temp)
# print(y.add(x))  # y不会变
# print(y.add_(x))  # y会变

# index
# x = torch.rand(3, 4)
# print(x)
# print(x[1, :])  # 取第一行元素

# 区别
# x = torch.randn(3, 4)  # 正态分布
# y = torch.rand(3, 4)  # 均匀分布
# print(x)
# print(y)

# 修改维度 view函数，将多维矩阵重新排成一维矩阵，再按要求重新排列形状。乘积需一样
# x = torch.randn(4, 4, 3)
# y = x.view(48)
# print(x)
# print(y)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())

# 将一个Torch Tensor转换为一个NumPy Array
# a = torch.ones(5)
# b = a.numpy()
# print(a)
# print(b)
# a = a.add_(1)
# print(b)  # 此时b相当于已经和a共用内存了，所以a变了b也变

# 将一个NumPy Array转换为一个Torch Tensor
# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)