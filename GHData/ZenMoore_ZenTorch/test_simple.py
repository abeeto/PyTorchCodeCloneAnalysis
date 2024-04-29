import torch
import torch.nn as nn
import numpy as np


# 构建输入集
x = np.mat('0 0')
x = torch.tensor(x).float()

# 搭建网络
myNet = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print(myNet)

myNet.load_state_dict(torch.load('param.pkl'))

# myNet = torch.load('model.pkl')

print(myNet(x).data)
