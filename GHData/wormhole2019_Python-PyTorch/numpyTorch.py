import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

tensor2array = torch_data.numpy()
print(
    '\n numpy',np_data,
    '\n torch', torch_data,
    '\n tensor2array', tensor2array,
)

#https://pytorch.org/docs/stable/torch.html#math-operations
#abs
data = -np_data
tensor = torch.FloatTensor(data)

print(
    '\n origin', data,
    '\n abs numpy',np.abs(data),
    '\n abs torch', torch.abs(tensor),
)

print(
    '\n origin', data,
    '\n sin numpy',np.sin(data),
    '\n sin torch', torch.sin(tensor),
)

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data = np.array(data)
print(
    '\n numpy',np.matmul(data,data),
    '\n torch', torch.mm(tensor,tensor),
)
# However, dot in torch is not useful
print(
    '\n numpy',np.dot(data,data),
    '\n torch', torch.dot(tensor[0],tensor[0]), # torch 会转换成 [1,2].dot([1,2) = 5.0
)
