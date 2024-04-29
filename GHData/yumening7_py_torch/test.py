'''
@author: Dzh
@date: 2019/11/29 10:37
@file: test.py
'''
# import numpy as np
# import torch

# np_data = np.arange(6).reshape((2, 3))
# # 创建tensor类型的数据
# torch_data = torch.from_numpy(np_data)
# # tensor类型转numpy类型
# tensor2array = torch_data.numpy()
#
# print(
#     '\nnumpy', np_data,
#     '\ntorch', torch_data,
#     '\ntensor2array', tensor2array
# )
# print(np_data)
# print(torch_data)

# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)     # 32bit
#
# # 取绝对值
# print(
#     '\nabs'
#     '\nnumpy', np.abs(data),
#     '\ntorch', torch.abs(tensor),
# )

# # 取sin值
# print(
#     '\nsin'
#     '\nnumpy', np.sin(data),
#     '\ntorch', torch.sin(tensor),
# )

# # 取平均值
# print(
#     '\nmean'
#     '\nnumpy', np.mean(data),
#     '\ntorch', torch.mean(tensor),
# )

# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)     # 32bit
#
# # 矩阵相乘
# print(
#     '\nnumpy', np.matmul(data, data),
#     '\ntorch', torch.matmul(tensor, tensor),
# )
requires_grad
# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)     # 32bit
# data = np.array(data)
#
# # dot相乘
# print(
#     '\nnumpy', data.dot(data),
#     '\ntorch', tensor.dot(tensor)
# )

import torch
from torch.autograd import Variable

data = [[1, 2], [3, 4]]

tensor = torch.FloatTensor(data)

variable = Variable(tensor, requires_grad=True)

print('tensor=', tensor)
print('variable=', variable)

t_out = torch.mean(tensor*tensor)   # x^2
v_out = torch.mean(variable*variable)

print('t_out=', t_out)
print('v_out=', v_out)

# 误差反向传递
v_out.backward()

# v_out = 1/4 * sum(var*var)
# d(v_out)/d(var) = 1/4 * 2 * variable = variable/2
print('grad=', variable.grad)

print('data=', variable.data)

print('data_numpy=', variable.data.numpy())