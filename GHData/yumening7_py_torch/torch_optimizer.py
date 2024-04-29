'''
@author: Dzh
@date: 2019/12/9 18:27
@file: torch_optimizer.py
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

# 学习率
LR = 0.01
# 每一批训练多少个数据
BATCH_SIZE = 32
# 总共迭代训练多少次
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# plt.scatter(x, y, c='b')
# plt.show()

# 把x和y转为TensorDataset类型
torch_dataset = Data.TensorDataset(x, y)
# 创建DataLoader用做数据分批训练
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 创建一个三层的神经网络模型
class Net(torch.nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)
        return x

# 分别创建模型对象
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
net_list = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# SGD 就是随机梯度下降
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
# momentum 动量加速,在SGD函数里指定momentum的值即可
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
opt_list = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

# 定义损失函数为均方差
loss_func = torch.nn.MSELoss()
# 定义loss_list用于接收不同optimizer
loss_list = [[], [], [], []]

for epoch in range(EPOCH):
    for setp, (x_batch, y_batch) in enumerate(loader):
        # 同时迭代三个list
        for net, opt, loss in zip(net_list, opt_list, loss_list):
            # 把x_batch传入net得出预测值
            predict = net(x_batch)
            # 使用损失函数得出当前loss
            current_loss = loss_func(predict, y_batch)
            # 优化器梯度归零
            opt.zero_grad()
            # 当前loss进行误差反向传递
            current_loss.backward()
            # 将参数更新值施加到 net 的 parameters 上
            opt.step()
            # 把当前loss添加到loss_list的指定位置中
            loss.append(current_loss.data.numpy())

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

# 循环绘图，y是对应的loss值，x是自动添加的步长
for i, loss in enumerate(loss_list):
    plt.plot(loss, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0, 0.2)
plt.show()