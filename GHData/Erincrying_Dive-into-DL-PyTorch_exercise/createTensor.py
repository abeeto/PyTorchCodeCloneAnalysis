# Tensor的最基本功能，即Tensor的创建
import torch

# 创建一个5x3的未初始化的Tensor
x = torch.empty(5, 3)
print(x, '5x3的未初始化的Tensor')

# 创建一个5x3的随机初始化的Tensor
x1 = torch.rand(5, 3)
print(x1, '5x3的随机初始化的Tensor')

# 创建一个5x3的long型全0的Tensor
x3 = torch.zeros(5, 3, dtype=torch.long)
print(x3, '5x3的long型全0的Tensor')

# 直接根据数据创建
x4 = torch.tensor([5.5, 3])
print(x4, '直接根据数据创建')

# 通过现有的Tensor来创建，此方法会默认重用输入Tensor的一些属性，例如数据类型，除非自定义数据类型。
x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x, '自定义数据类型')

x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
print(x, '指定新的数据类型') 

# 通过shape或者size()来获取Tensor的形状
print(x.size(), 'x.size')
print(x.shape, 'x.shape')