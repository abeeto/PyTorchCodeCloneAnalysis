# -*- coding: utf-8 -*-
from __future__ import print_function
import Base
import torch

Base.printT("========== Tensor ==========")
# 构建未初始化的5x3矩阵
# printC("构建未初始化的5x3矩阵")
# x = torch.Tensor(5, 3)
# x = torch.empty(5, 3)
# print(x)

# 构造一个随机初始化的矩阵
Base.printC("构造一个随机初始化的矩阵")
x = torch.rand(5, 3)
print(x)

# 构造矩阵填充零和dtype long
Base.printC("构造矩阵填充零和dtype long")
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接从数据构造张量
Base.printC("直接从数据构造张量")
x = torch.tensor([5.5, 3, 5.5])
print(x)

# 或根据现有张量创建张量。这些方法将重用输入张量的属性，例如dtype，除非用户提供了新的值
Base.printC("或根据现有张量创建张量。这些方法将重用输入张量的属性，例如dtype，除非用户提供了新的值")
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# 获取其大小
Base.printC("获取其大小")
print(x.size())

Base.printT("========== 操作 ==========")

# 增加 语法1
Base.printC("增加 语法1")
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)

# 增加 语法2
Base.printC("增加 语法2")
print(torch.add(x, y))

# 增加：提供输出张量作为参数
Base.printC("增加：提供输出张量作为参数")
result = torch.zeros(5, 3)
print(result)
torch.add(x, y, out=result)
print(result)

# 增加：in-place
# Any operation that mutates a tensor in-place is post-fixed with an _.
# For example: x.copy_(y), x.t_(), will change x.
Base.printC("增加：in-place")
y.add_(x)
print(y)

# You can use standard NumPy-like indexing with all bells and whistles!
Base.printC("You can use standard NumPy-like indexing with all bells and whistles!")
print(x[:, 1])

# 调整大小：如果要调整大小/重建张量，可以使用torch.view：
# rand是生成均匀分布，而randn是生成均值为0，方差为1的正态分布
Base.printC("调整大小：如果要调整大小/重建张量，可以使用torch.view：")
x = torch.randn(4, 4)
print(x.size())
y = x.view(16)
print(y.size())
z = x.view(-1, 8)  # 大小-1是从其他维度推断出来的
print(z.size())

# 如果你有一个单元张量，用.item()作为一个Python数字来获取它的值
Base.printC("如果你有一个单元张量，用.item()作为一个Python数字来获取它的值")
x = torch.randn(1)
print(x)
print(x.item())

Base.printT("========== NumPy ==========")
import numpy

# 将torch张量转换为NumPy阵列，反之亦然
Base.printC("将torch张量转换为NumPy阵列，反之亦然")
a = torch.ones(5, dtype=torch.long)
b = a.numpy()
print(a)
print(b)

# 看看numpy数组如何改变值
Base.printC("看看numpy数组如何改变值")
a.add_(1)
print(a)
print(b)

# 将NumPy数组转换为torch张量
Base.printC("将NumPy数组转换为torch张量")
a = numpy.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
numpy.add(a, 1, out=a)
print(a)
print(b)
# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.


Base.printT("========== cuda ==========")
###############################################################
# All the Tensors on the CPU except a CharTensor support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# Tensors can be moved onto any device using the ``.to`` method.

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!
