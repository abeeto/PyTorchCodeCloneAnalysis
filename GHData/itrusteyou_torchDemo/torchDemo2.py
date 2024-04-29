import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as tn
import torch.nn.functional as F
from IPython import display
import random

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# #y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# y_data = torch.Tensor([[0], [0], [1]])
#
# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = F.signoid(self.linear(x))
#         return y_pred
#
# model = LogisticRegressionModel()
#
# class LinearModel(tn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = tn.Linear(1, 1)
#
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
#
# #model = LinearModel()
#
# criterion = torch.nn.BCELoss(size_average=False)
# #criterion = tn.MSELoss(size_average=False)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#
# for epoch in range(1000):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())
#
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred ', y_pred.data)

# x = torch.tensor([5.5, 3])
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
# print(x)

torch.cuda.manual_seed(10)

lr = 0.05  # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
# torch.randn(20, 1) 用于添加噪声
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True) # 设置梯度求解为 true
b = torch.zeros((1), requires_grad=True) # 设置梯度求解为 true

# 迭代训练 1000 次
for iteration in range(1000):

    # 前向传播，计算预测值
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 每次更新参数之后，都要清零张量的梯度
    w.grad.zero_()
    b.grad.zero_()

    # 绘图，每隔 20 次重新绘制直线
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
      

        # 如果 MSE 小于 1，则停止训练
        if loss.data.numpy() < 1:
            break