import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x),Variable(y)

def save():
    # 构建神经网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 定义优化方式（随机梯度下降）
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.1)

    # 定义损失函数计算方式
    loss_func = torch.nn.MSELoss()

    for i in range(1000):
        prediction = net1(x) # 得到当前结果
        loss = loss_func(prediction, y) # 计算当前损失
        optimizer.zero_grad()  # 梯度设为零
        loss.backward()  # 反向传递
        optimizer.step()  # 优化梯度

    torch.save(net1,'net.pkl') # 保存整个网络
    torch.save(net1.state_dict(),'net_params.pkl')# 只保存参数

    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Netl')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw = 5)


def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.figure(1,figsize=(10,3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw = 5)

def restore_params():
    net3 = net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    plt.figure(1,figsize=(10,3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw = 5)

save()

restore_net()

restore_params()

plt.show()