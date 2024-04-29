import numpy as np
import torch

a = np.array([2, 3.3])
a_tensor = torch.from_numpy(a)

a = np.ones([2, 3])
a_tensor = torch.from_numpy(a)
print(a_tensor)

a = torch.tensor([2, 3.2])
print(a)

a = torch.FloatTensor([2., 3.2])
print(a)

a = torch.FloatTensor(2, 3) #生成两行三列的随机数
print(a)

a = torch.tensor([[2, 2.3], [3, 2.3]])
print(a)

#设置默认的数据类型
print(torch.tensor([1.2, 3]).type())

torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type())

# rand/rand_like, randint

a = torch.rand(3, 3)
print(a)
b = torch.rand_like(a)
print(b)
c = torch.randint(1, 10, (3, 3))
print(c)

#生成一个正态分布的
torch.randn(3, 3)

torch.normal(mean=0.0, std=1.0, size=(3, 3))

#全部赋值为一个数
a = torch.full([2, 3], 7)
print(a)

b = torch.full([], 7)

c = torch.full([1], 7)
print(c)

#生成递增等差数列
a = torch.arange(0, 10)

#linspace, logspace
torch.linspace(0, 10, steps=4)

torch.logspace(0, -1, steps=11)

#Ones/zeors/eye
torch.ones(3, 3)

torch.zeros(3, 3)

c = torch.eye(3, 4)
print(c)

#randperm 生成随机的索引值
a = torch.randperm(10)
print(a)



