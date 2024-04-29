import torch
import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
#
# print(
#     '\nnumpy', np_data,
#     '\ntorch', torch_data,
#     '\ntensor2array', tensor2array
# )

# abs
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 32bit
#
# print(
#     '\nmean',
#     '\nnumpy', np.mean(data),      # [1 2 1 2]
#     '\ntorch', torch.mean(tensor)  # [1 2 1 2]
# )

# 矩阵
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point

print(
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor)
)
