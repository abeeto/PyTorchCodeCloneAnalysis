# 广播机制

# 当对两个形状不同的Tensor按元素运算时，
# 可能会触发广播（broadcasting）机制：先适当复制元素使这两个Tensor形状相同后再按元素运算
import torch

x = torch.arange(1, 3).view(1, 2)
print(x, 'x')
y = torch.arange(1, 4).view(3, 1)
print(y, 'y')
print(x + y, 'x+y')