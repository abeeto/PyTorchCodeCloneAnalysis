# 本节课学习一下激励函数

import torch
import torch.nn.functional as F 
from torch.autograd import Variable 
import matplotlib.pyplot as plt

x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
#y_softmax = torch.softmax(x).data.numpy()
# y_softmax = F.softmax(x)

print(y_relu)
print(y_sigmoid)
print(y_tanh)
