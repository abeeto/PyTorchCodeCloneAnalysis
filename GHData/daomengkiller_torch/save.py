import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 产生100×1的数据，一维数据-1到1，中间有100个数据，重新打乱数据
y = x.pow(2) + 0.2 * torch.rand(x.size())  # 计算每个数据的方程

x, y = Variable(x), Variable(y)  # 变成容器变量


# plt.scatter(x.data.numpy(), y.data.numpy())  # 显示出来离散
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


# 训练神经网络
def train(x, y):
    # mynet = Net(n_feature=1, n_hidden=10, n_output=1)
    mynet = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))#此处默认激励函数也为一层
    print(mynet)

    optimizer = torch.optim.SGD(mynet.parameters(), lr=0.5)#设置梯度下降函数
    loss_func = torch.nn.MSELoss()
    plt.ion()
    for t in range(100):
        prediction = mynet(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'iter=%d\nLoss=%.4f' % (t, loss.data[0]), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    # plt.show()
    return mynet


# saving the net
def save(net):
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')


def restore_net():
    # 第一种方法保存神经网络，直接保存图结构
    net2 = torch.load('net.pkl')
    # 第二种方法保存神经网络，只是保存权重，需要重新构建图结构
    net3 = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    net3.load_state_dict(torch.load('net_params.pkl'))
    return net2, net3


def plot(x, y, prediction, n, i):
    plt.figure(1, figsize=(10, 3))
    plt.subplot(1, n, i)
    plt.title('net%d' % i)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


net1 = train(x, y)
save(net1)
net2, net3 = restore_net()

plot(x, y, net1(x), n=3, i=1)
plot(x, y, net2(x), n=3, i=2)
plot(x, y, net3(x), n=3, i=3)
plt.show()
