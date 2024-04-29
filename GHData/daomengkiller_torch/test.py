import torch
from torch.autograd import Variable  # 导入
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 产生100×1的数据，一维数据-1到1，中间有100个数据，重新打乱数据
y = x.pow(2) + 0.2 * torch.rand(x.size())  # 计算每个数据的方程

x, y = Variable(x), Variable(y)  # 变成容器变量


# plt.scatter(x.data.numpy(), y.data.numpy())  # 显示出来离散
# plt.show()  # 显示


class Net(torch.nn.Module):  # 生成神经网络
    def __init__(self, n_feature, n_hidden, n_output):  # 定义特征输入矩阵，隐藏层，结果输出层
        super(Net, self).__init__()  # 标准的初始化
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 此处认为数据矩阵也是一层，这里设置隐藏层的输入和输出，即隐藏层为hidden
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 这里的linear为线性的函数

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 激励函数，是切分的功能，而且可以使切分弯曲
        x = self.predict(x)  # 预测
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # 生成网络
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 摄制梯度的优化方法，即下降的方法
loss_func = torch.nn.MSELoss()  # 设置平方和最小的损失函数。
plt.ion()  # 打开交互功能，为图标
for t in range(500):  # 训练500次
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  #
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
