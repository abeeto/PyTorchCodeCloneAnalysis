import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt  # python 的可视化模块


#*************************************************

#  numpy ----> tensor  : .from_numpy
#  tensor ----> numpy : .numpy
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy',np_data,
    '\nnumpy',torch_data,
    '\ntensor2array', tensor2array,
)
#*************************************************


data = [-1,-2,2,2]
tensor = torch.FloatTensor(data)
print(
    '\nsin', np.sin(np_data),
    '\nnumpysin', np.sin(np_data),
    '\ntorchsin', torch.sin(tensor),
)

#*************************************************

data = [[1,2],[3,4]]
tensor_data = torch.FloatTensor(data)
# 转化为数组
data = np.array(data)
print(
    '\nnumpy', data.dot(data),
    '\ntorch', tensor_data.mm(tensor_data),
)

#*************************************************

# Variable ----> numpy : .data.numpy
# var.grad : function gradient

tensor = torch.FloatTensor([[8,2],[3,4]])
var = Variable(tensor,requires_grad = True)
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(var * var)
print(t_out)
print(v_out)
v_out.backward()
print(var)
print(var.data)
print("print the gradient \n")
print(var.grad)
print(var.data.numpy)

#*************************************************

x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()   # 换成 numpy array, 出图时用

# 几种常用的 激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)  softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

#神经网络优化器优化的对象是网络的参数，因此优化器的输入参数是网络的参数







