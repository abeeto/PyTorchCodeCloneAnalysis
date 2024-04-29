import torch

print(torch.cuda.is_available())  # 检查gpu是否可用
# torch.Tensor是存储和变换数据的主要工具,Tensor提供GPU计算和自动求梯度等更多功能
# "tensor"这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。
# x = torch.empty(5, 3)  # 创建一个5x3的未初始化的Tensor
# print()
# x = torch.rand(5, 3)  # 创建一个5x3的随机初始化的Tensor
# print(x)
# x = torch.zeros(5, 3, dtype=torch.long)  # 创建一个5x3的long型全0的Tensor
# print(x)
# x = torch.tensor([5.5, 3])  # 直接根据数据创建
# print(x)
# print(x.size())  # 可以通过shape或者size()来获取Tensor的形状:
# print(x.shape)
#
# # 以下代码只有在PyTorch GPU版本上才会执行
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # GPU
#     y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
#     x = x.to(device)                       # 等价于 .to("cuda")
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型

a = torch.ones(3)
b = 10
print(a + b)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


