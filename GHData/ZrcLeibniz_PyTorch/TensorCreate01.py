import torch
import numpy

# 创建tensor的第一种方式
n_a = numpy.array([2, 3.3])
t_a = torch.from_numpy(n_a)
print(t_a)

# 创建tensor的第二种方式
t_a1 = torch.tensor([2., 3.2])
print(t_a1)
t_a2 = torch.tensor([[2., 3.2], [1., 2.768]])
print(t_a2)

# uninitialized
empty = torch.empty(1)
print(empty)
tensor = torch.Tensor(2, 3)
print(tensor)
int_tensor = torch.IntTensor(2, 3)
print(int_tensor)
float_tensor = torch.FloatTensor(2, 3)
print(float_tensor)


