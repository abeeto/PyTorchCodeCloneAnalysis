
# Tensor_arithmetic_operation,算术运算:

import torch
a = torch.tensor((2,2),dtype=torch.float32)
b = torch.tensor((2,2),dtype=torch.float32)
print(a)
print(b)

'''
# # 加法运算。  减法运算
# c = a + b
# c = torch.add(a,b)
# a.add(b)
# a.add_(b) # a值更新为a+b的值
'''
# print(a+b)
# print(torch.add(a,b))
# print(a.add(b))
# print(a)
# print(a.add_(b))
# print(a)


'''
# # 乘法运算,哈达玛积，即对应元素的相乘,点积。  # # 除法运算，/,div()
# c = a * b
# c = torch.mul(a,b)
# a.mul(b)
# a.mul_(b)
'''
# print(a*b)
# print(torch.mul(a,b))
# print(a.mul(b))
# print(a.mul_(b))
# print(a)


'''
# 矩阵运算：对于高维的Tensor(dim>2),定义其矩阵乘法仅在最后的两个维度上，要求前面的维度必须保持一致，就像矩阵的索引一样并且运算只有torch.matmul()
# 以下为例,a的前两个为1,2,而b的前两个必须也为1,2
# a = torch.ones(1,2,3,4)
# b = torch.ones(1,2,4,3)
# print(a.matmul(b))
# print(torch.matmul(a,b))
'''
# c = torch.ones(1,2,3,4)
# d = torch.ones(1,2,4,3)
# print(c.matmul(d).shape)


'''
# 幂运算
# print(torch.pow(a,2))
# print(a.pow(2))
# print(a**2)
# print(a.pow_(2))
# # e的n次方
# print(torch.exp(a))
# b = a.exp_()
'''



'''
# 开方运算
# print(a.sqrt())
'''

'''
# 对数运算
# print(torch.log2(a)) # 底数为2
# print(torch.log10(a)) # 底数为10
# print(torch.log(a)) # 底数为e
# print(torch.log_(a))
'''

'''
# Tensor的取整/取余运算
# .floor()向下取整数
# .ceil()向上取整数
# .round()四舍五入
# .trunc()裁剪，只取整数部分
# .frac()只取小数部分
# %取余
'''




