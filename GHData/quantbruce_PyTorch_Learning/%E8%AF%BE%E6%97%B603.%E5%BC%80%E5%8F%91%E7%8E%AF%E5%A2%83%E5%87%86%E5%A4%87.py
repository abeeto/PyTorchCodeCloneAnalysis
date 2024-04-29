#########pyTorch 矩阵运算

import torch
import time


print(torch.__version__)
print(torch.cuda.is_available())
print(torch._C._cuda_getDeviceCount())


a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

# CPU模式
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1-t0, c.norm(2))

# GPU模式
device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1-t0, c.norm(2))


# 再次运行GPU模式，速度明显加快
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))



############ 自动求导(偏导数)
import torch
from torch import autograd

x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**3*x + b**2*x + c*x
print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print(grads)
print('after:', grads[0], grads[1], grads[2])

