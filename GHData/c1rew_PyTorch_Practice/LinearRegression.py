import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# y = 5.2x+3  一元线性函数，斜率 5.2，截距3
true_w = 5.2
true_b = 3
x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)
y = true_w*x + true_b

# 为y值增加高斯噪音，弄得像随机的，等下拟合出一条直线来回归这些点
y = y + torch.randn(x.size()) 

# 打印散点图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# plt更新
plt.ion()
plt.show()

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 线性模型，一个输入，一个输出
    
    def forward(self, x):
        return self.linear(x)
    
model = LinearRegression()

# 均方误差损失
loss_function = torch.nn.MSELoss()

# 随机梯度下降，学习率 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 另外一个优化方法 Adam
#optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

# 迭代次数 2000
epochs_num = 2000

for epoch in range(epochs_num):
    # forward
    out = model(x)
    loss = loss_function(out, y)
    
    # backward
    optimizer.zero_grad()  # 梯度还原为零
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    if (epoch+1) % 20 == 0:
        # print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,epochs_num,loss.data.item()))
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), out.data.numpy(), 'r-', lw=5)
        plt.text(-2, 12, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':15, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
