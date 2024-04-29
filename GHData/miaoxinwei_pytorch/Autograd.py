# -*- coding: utf-8 -*-
from __future__ import print_function
import Base
import torch

Base.printT("========== Tensor ==========")

# 创建一个张量并设置requires_grad = True来跟踪计算
Base.printC("创建一个张量并设置requires_grad = True来跟踪计算")
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 做张量的操作
Base.printC("做张量的操作")
y = x + 2
print(y)

# y是由于操作而创建的，所以它有一个grad_fn
Base.printC("y是由于操作而创建的，所以它有一个grad_fn")
print(y.grad_fn)

# 对y做更多的操作
Base.printC("对y做更多的操作")
z = y * y * 3
out = z.mean()
print("z: ", z, " \nout: ", out)

# .requires_grad_（...）就地更改现有张量的requires_grad标志。 如果没有给出，输入标志默认为True。
Base.printC(".requires_grad_（...）就地更改现有张量的requires_grad标志。 如果没有给出，输入标志默认为True。")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
# a.requires_grad_()
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

Base.printT("========== Gradients ==========")

# 现在让我们来回顾一下out包含单个标量，out.backward()相当于out.backward(torch.tensor(1))。
Base.printC("现在让我们来回顾一下out包含单个标量，out.backward()相当于out.backward(torch.tensor(1))。")
out.backward()

# 打印Gradients d(out)/ dx
# 我们称out为张量“o”。我们有o=1/4∑izi，zi=3(xi+2)(xi+2) 和 zi∣xi1=27。因此，∂o/∂xi=(xi+2)*3/2，因此 ∂o∂xi∣xi1=9/2=4.5。
Base.printC("打印Gradients d(out)/ dx")
Base.printC("我们称out为张量“o”。我们有o=1/4∑izi，zi=3(xi+2)(xi+2) 和 zi∣xi1=27。因此，∂o/∂xi=(xi+2)*3/2，因此 ∂o∂xi∣xi1=9/2=4.5。")
print(x)
print(x.grad)
