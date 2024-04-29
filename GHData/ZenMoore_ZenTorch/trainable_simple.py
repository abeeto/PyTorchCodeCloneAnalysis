import torch
import torch.nn as nn
import numpy as np

# 求解异或运算

# 构建输入集
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()

# 搭建网络
myNet = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print(myNet)

# 设置优化器
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
loss_func = nn.MSELoss()

for epoch in range(5000):
    out = myNet(x)
    loss = loss_func(out, y)  # 计算误差
    optimzer.zero_grad()  # 清除梯度
    loss.backward()
    optimzer.step()
    print(loss.data)

# torch.save(myNet, 'model.pkl')
torch.save(myNet.state_dict(), 'param.pkl')