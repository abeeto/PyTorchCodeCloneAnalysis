# -*- coding: utf-8 -*-
# @Time : 2022/4/10 15:21
# @Author : hhq
# @File : pytorch_test1.py
import torch
import numpy as np

# todo 指定类型
x1 = torch.zeros(5, 3, dtype=torch.long)
x2 = torch.zeros(5, 3).long()
# print(x1 == x2)

# todo 根据已有数据创建tensor
y1 = torch.tensor([5.5, 3])
# todo 从一个已有的tensor建立新的tensor，新建的tensor会有之前的tensor的一些特征
y2 = y1.new_ones(5, 3)
# print(y1)
# print("y2=", y2)
# 以上两个tensor数据类型相同
# todo 指定类型
y3 = y1.new_ones(5, 3, dtype=torch.double)
# print(y3)
# todo 产生与原tensor形状相同的随机tensor
z = torch.randn_like(y2, dtype=torch.float)
# print(z)
# print(z.shape)
# tensor([-1.0115, -0.2393])
# torch.Size([2])
# todo tensor的加法
# print(torch.rand(5,3)+torch.rand(5,3))
# print(torch.add(x2,y2))
# print(x2.add_(y2)) # 相当于x2 = x2+y2
# in-place形式的加法：像这种有下划线的方法，都会改变调用者原有的值。
# todo numpy的切片同样适用tensor
# print(z[:, 1:])
# todo 利用view()函数达到numpy的reshape作用
a = torch.rand(4, 4)
# b = a.view()
c = a.view(16, )
d = a.view(16)
e = a.view(16, 1)
f = a.view(2, 8)
# todo 若只有一个元素的tensor，可以使用.item()转化为一个数值
p = torch.randn(1)
# print(p)
# print(p.item())
# todo 转置
# print(f.transpose(0,1))


# todo numpy和tensor之间相互转化,tensor与numpy可相互转化，并且共同使用一个内存空间
ff = f.numpy()
# print(ff)
gg = np.ones(5)
# print(gg)
g = torch.from_numpy(gg)
# print(g)


# todo GPU训练
# 使用.to方法，Tensor可以被移动到别的device上
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)
    y = torch.ones_like(y2, device=device)
    y2 = y2.to(device)
    zz = y + y2
    # print(zz)  # 显示cuda
    print(zz.to("cpu", torch.double))  # 不显示cuda
    # print("")
    # todo numpy是cpu的库，一定要tensor从gpu搬到cpu上才能转为numpy
    # print("b",y)
    # print(y.numpy())  # 报错：can't convert cuda:0 device type tensor to numpy.
    # Use Tensor.cpu() to copy the tensor to host memory first.
    # y = y.to("cpu").data.numpy()
    y = y.cpu().data.numpy()  # 与上一行作用相同
    print(y)

# todo numpy.ones_like
# numpy.ones_like(a,dtype=None,order='K',subok=True)
# 返回指定数组具有相同形状和数据类型的数组，并且数组中值都为1
