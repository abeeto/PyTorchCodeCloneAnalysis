import torch
import numpy

# 生成一个随机的tensor维度的2行3列,服从正态分布
a = torch.randn(2, 3)

# 获得tensor的类型
print(a.type())

# 验证一个数据是否合法
print(isinstance(a, torch.FloatTensor))

# ------------------------------------------------------------------------------------------

# Dimension 0 / rank 0
# 在神经网络中经常用作损失函数
b = torch.tensor(1.)
print(b)

c = torch.tensor(1.3)
print(c)

# 获得tensor的shape
print(c.shape)
print(a.shape)

# -------------------------------------------------------------------------------------------

# Dimension 1 / rank 1
# 在神经网络中经常用作bias
# 也可以用作Linear Input
e = torch.tensor([1.1])
print(e)

f = torch.tensor([2.2, 1.1])
print(f)
# 使用numpy产生数据，之后转换为tensor

data = numpy.ones(2)
data2 = torch.from_numpy(data)
print(data2)

# ------------------------------------------------------------------------------------------

# Dimension2
g = torch.randn(3, 4)
print(g.shape)
print(g.shape[0])
print(g.shape[1])

# -------------------------------------------------------------------------------------------

# Dimension3
# 使用场景在RNN中
h = torch.rand(2, 2, 3)
print(h)
print(h.shape[0])
print(h.shape[1])
print(h.shape[2])
# 可以将tensor的维度转换为一个list
print(list(h.shape))

# Dimension4
# 使用场景在输入的图片就可以表示为4维tensor
i = torch.rand(2, 1, 28, 28)
print(i)
print(list(i.shape))

# 可以得到某一个tensor具体占用的内存的大小
print(i.numel())

# 可以得到一个tensor的维度
print(i.dim())
