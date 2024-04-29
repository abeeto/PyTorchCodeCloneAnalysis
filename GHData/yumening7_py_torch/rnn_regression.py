'''
@author: Dzh
@date: 2020/1/9 10:40
@file: rnn_regression.py
'''

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

EPOCH = 100
INPUT_SIZE = 1      # 输入
TIME_STEP = 10      # 时间点的数量
LR = 0.02           # 学习率
HIDDEN_SIZE = 32    # 隐藏层神经元数量

'''
这是一个RNN回归的Demo，目的是用sin做为特征值，去预测对应的cos
'''
data = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
x_np = np.sin(data)
y_np = np.cos(data)

plt.plot(data, x_np, 'b', label='target(sin)')
plt.plot(data, y_np, 'r', label='target(cos)')
plt.legend(loc='best')
plt.show()

# print(data)

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,          # 输入大小
            hidden_size=HIDDEN_SIZE,        # 隐藏层神经元数量
            num_layers=1,                   # RNN神经网络层数
            batch_first=True                # batch是否在第一维度
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)    # 把RNN结果放入BP神经网络

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))           # 把每个时间点的结果输出都传入self.out
        return torch.stack(outs, dim=1), h_state

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(EPOCH):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r')                        # 拷贝原始数据
    plt.plot(steps, prediction.data.numpy().flatten(), 'b')
    plt.draw()              # 连续绘制
    plt.pause(0.01)         # 绘制间隔

plt.ioff()
plt.show()