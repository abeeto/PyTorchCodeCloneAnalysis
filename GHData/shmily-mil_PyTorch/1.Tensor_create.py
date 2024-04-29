import torch

print('='*80)
print('使用Tensor(data)的方式创建Tensor:')
a = torch.Tensor([[1,2],[3,4]])
print(a)
print(a.type())

print('='*80)
print('使用Tensor(shape)的方式创建Tensor:')
b = torch.Tensor(2,3)
print(f'打印b的值为：{b}')
print(f"打印b的类型为：{b.type()}")

print('='*80)
print('创建全是1的Tensor:')
c = torch.ones(2,2)
print(f"打印c的值为:{c}")
print(f"打印c的类型为:{c.type()}")

print('='*80)
print('创建全是0的Tensor:')
c = torch.zeros(2,2)
print(f"打印c的值为:{c}")
print(f"打印c的类型为:{c.type()}")

print('='*80)
print('创建对角线为1的Tensor')
c = torch.eye(2,2)
print(f"打印c的值为:{c}")
print(f"打印c的类型为:{c.type()}")

print('='*80)
print('创建一个和d相同shape的Tensor')
d = torch.Tensor(4,4)
d = torch.zeros_like(d)
print(f'打印d的值为：{d}')

print('='*80)
print('创建一个和e相同shape的Tensor')
e = torch.Tensor(4,4)
e = torch.ones_like(e)
print(f'打印d的值为：{e}')

print('='*80)
print('创建随机值的Tensor')
a = torch.rand(2,2) # 生成的随机值是0-1之间的
print(a)
print(a.type())

print('='*80)
print('创建正态分布的Tensor')
a = torch.normal(mean=0.0,std=torch.rand(5))
print(a)
print(a.type())

print('='*80)
print('创建均匀分布的Tensor')
a = torch.Tensor(2,2).uniform_(-1,1)
print(a)
print(a.type())

print('='*80)
print('创建序列的Tensor')
a = torch.arange(0,10,1)
print(a)
print(a.type())

print('='*80)
print('创建等间隔的Tensor')
a = torch.linspace(2,10,3)
print(a)
print(a.type())

print('='*80)
print('生成打乱顺序的序列')
a = torch.randperm(10)
print(a)
print(a.type())

############################################
import numpy as np
a = np.array([[1,2],[2,2]])
print(a)









