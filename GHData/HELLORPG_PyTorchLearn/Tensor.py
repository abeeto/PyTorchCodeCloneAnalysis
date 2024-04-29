"""
这个文件内涉及一些张量（tensor）的基础运算，
用于熟悉PyTorch中对其的表述和应用。
"""

import torch

x = torch.ones(2, 2, requires_grad=True)
# 2*2的1矩阵，同时开启操作追踪（用于后续求解backward()）
# required_grad默认是False
print(x)

y = x + 2
print(y)

print(y.grad_fn)
# 对于所有的计算结果，都有 grad_fn 属性


z = y * y * 3
print(z, z.mean())
