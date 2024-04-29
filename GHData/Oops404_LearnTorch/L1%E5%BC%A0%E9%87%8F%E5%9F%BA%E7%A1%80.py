import torch
import numpy as np

t1 = torch.tensor([1, 2])
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 查看维度
print(t2.ndimension())
# 查看形状
print(t2.shape)
# 形变
print(t2.reshape(3, 2))
print(t2.size())

# 最底层包含多少个元素
print(t2.numel())

# len的含义,注意理解，顶层对象的个数
print(len(t2))
# 按行排列，拉平，二向箔到1纬
print(t2.flatten())
# ----------- 0维 张量----------
# 也可以认为是标量
t0 = torch.tensor(1)
# 纬度数
print(t0.ndimension())
print(t0.shape)
print(t0.numel())

# 特殊张量
# 创建3行2列 全0张量
print(torch.zeros([3, 2]))
# 全1张量
print(torch.ones([2, 3]))
# 单位举证
print(torch.eye(5))

# 对角线矩阵
td = torch.tensor([1, 3, 5, 7, 9, 11])
print(torch.diag(td))

# 随机数张量
print(torch.rand([2, 3]))
# 随机数张量，符合标准正态分布 n:normal
print(torch.randn([2, 3]))

# 均值为2，标准差为3的张量
print(torch.normal(2, 3, size=(2, 2)))

# 在范围之内随机选择整数
print(torch.randint(1, 10, size=(2, 2)))

# 生成数列张量
print(torch.arange(10))
# 步长为0.5，生成数列
print(torch.arange(1, 10, 0.5))
# 等距离为2，选择数，生成数列
print(torch.linspace(1, 10, 3))

# 未初始化3行2列，但是已经指定形状的矩阵
print(torch.empty(3, 2))

# 根据指定形状2行4列，填充指定类型的数值6
print(torch.full([2, 4], fill_value=6, dtype=torch.int16))

# _like 依据某个张量的形状,有很多带有_like的含义相同
# full_like，填充指定数值，形成新的张量
print(torch.full_like(td, fill_value=6, dtype=torch.int16))

print(td.numpy())
print(td.tolist())

# 0维张量转化成数值
print(t0.item())


