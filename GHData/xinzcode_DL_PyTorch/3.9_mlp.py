import torch
from torch import nn
from torch.nn import init
import numpy as np
from d2lzh_pytorch.utils import *

num_inputs, num_outputs, num_hiddens = 784, 10, 256
batch_size = 256
num_epochs = 5

# 准备数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义模型
net = nn.Sequential(
        FlattenLayer(),  # 转换输入形状
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
        )

# 定义参数
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)  # 用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数

# W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)  # 第一层权重
# b1 = torch.zeros(num_hiddens, dtype=torch.float)  # 第一层偏差
# W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)  # 第二层权重
# b2 = torch.zeros(num_outputs, dtype=torch.float)  # 第二层偏差
# params = [W1, b1, W2, b2]
# for param in params:
#     param.requires_grad_(requires_grad=True)  # 开始跟踪这个tensor上面的所有运算，
#     # 如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
#
# # 激活函数  ReLU函数只保留正数元素，并将负数元素清零
# def relu(X):
#     return torch.max(input=X, other=torch.tensor(0.0))
#
# def net(X):
#     X = X.view((-1, num_inputs))        # 输入层
#     H = relu(torch.matmul(X, W1) + b1)  # 隐含层
#     return torch.matmul(H, W2) + b2     # 输出层


# 损失函数
loss = torch.nn.CrossEntropyLoss()

# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 训练模型
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# num_epochs, lr = 5, 100.0
# train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
