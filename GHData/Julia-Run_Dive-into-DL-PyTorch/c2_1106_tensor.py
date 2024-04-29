import torch

# a = torch.empty(5, 3)
# print(a)
#
# b = torch.rand(5, 3)
# print(b)
#
# c = torch.zeros(5, 3, dtype=torch.long)
# print(c)
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.float64)
print(x)
x = torch.rand_like(x, dtype=torch.float)
print(x)
print(x.size(), x.shape, sep="------")
x = torch.eye(3)
print(x)
x = torch.tensor([[1, 1, 3], [2, 2, 4]])
print(x)
x = torch.arange(1, 10, 3)
print(x)
x = torch.linspace(1, 10, 3)
print(x)
x = torch.rand(1, 3)
print(x)
x = torch.randn(1, 3)
print(x)
x = torch.randperm(10)
print(x)
## normal(), uniform()???


print(x)
a = torch.rand(5, 3)
b = torch.ones(5, 3)
x = a + b
print(x)
x = torch.add(a, b)
print(x)
a.add_(b)  # inplace操作 。yTorch操作inplace版本都有后缀_, 例如x.copy_(y), x.t_()
print(a)

y = x[4, :]
# y = y + 1  #这两种加法结果不一样
y += 1  # 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
print(x[4, :], y, sep='---x to y ---')
x = torch.eye(3)
y = torch.nonzero(x)
print(y)
y = torch.index_select(x, -2, torch.tensor([0, 1]))
print(y)
# view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也即更改其中的一个，另外一个也会跟着改变
x = torch.rand(5, 4)
y = x.view(-1)  # -1 依据原数组和现有维度，自动调整维度 （=  y=x.view(20)）
z = x.view(2, 2, -1)  # -1 --> 5
print(x.size(), y.size(), z.size())
print(x, y, z, sep='\n')

x_cp = x.clone().view(-1)
x += 1
print(x, y, x_cp, sep='\n')
t = torch.rand(1)
print(t)
print(t.item())  # item只能转变一维的tensor

x = torch.eye(3)
y = torch.trace(x)
z = torch.diag(x)
x1 = torch.tril(x)
print(y, z, x1, sep='\n')
x = torch.rand(3, 3)
x1 = torch.tril(x)
x2 = torch.triu(x)
y1 = torch.t(x1)
print(x1, x2, y1, sep='\n')

import numpy as np

a = torch.ones(5, 3)
b = a.numpy()
print(a, b, id(a), id(b), sep='\n')
a = np.ones(5)
b = torch.from_numpy(a)
c = id(b)
b += 1
print(a, b, id(a), id(b), c, sep='\n')
