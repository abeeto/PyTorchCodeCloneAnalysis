import torch
from torch.autograd import Variable  # 容器变量
import torch.nn.functional as F  #
import matplotlib.pyplot as plt  # 绘图工具

# x0和x1是作为输入的，直接在图像
n_data = torch.ones(100, 2)  # 产生100*2的一阵
x0 = torch.normal(2 * n_data, 1)  # 使数据矩阵，正态化分布。
y0 = torch.zeros(100)  # 产生100*1的零矩阵
x1 = torch.normal(-2 * n_data, 1)  # 使数据正太化
y1 = torch.ones(100)  # 产生100*1的矩阵

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 默认为竖向连接并接一起
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)  # 变成容器变量


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')  # 显示出来离散
# plt.show()  # 显示


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(n_feature=2, n_hidden=10, n_output=2)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 1 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'iter=%d\nAccuracy=%.2f' % (t + 1, accuracy), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
