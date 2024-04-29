'''
@author: Dzh
@date: 2019/12/9 10:13
@file: save_extraction.py
'''
import torch
from torch.nn import Sequential
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x = Variable(x)
y = Variable(y)

def save():
    net1 = Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(num=1, figsize=(10, 3))
    plt.subplot(131)
    plt.scatter(x.data.numpy(), y.data.numpy(), c='b')
    plt.plot(x.data.numpy(), prediction.data.numpy(), c='r', lw=5)
    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})

    # 保存整个神经网络模型，保存速度较慢
    torch.save(net1, 'net.pkl')
    # 只保存神经网络的参数配置，不保存层结构。保存速度较快
    torch.save(net1.state_dict(), 'net_params.pkl')

def restore_net():
    '''
    提取整个神经网络并绘图
    :return:
    '''
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.subplot(132)
    plt.scatter(x.data.numpy(), y.data.numpy(), c='b')
    plt.plot(x.data.numpy(), prediction.data.numpy(), c='red', lw=5)
    # plt.text(0.5, 1, 'Loss=%.4f'%)

def restore_net_params():
    '''
    再次定义层结构，提取神经网络参数并绘图
    :return:
    '''
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    plt.subplot(133)
    plt.scatter(x.data.numpy(), y.data.numpy(), c='b')
    plt.plot(x.data.numpy(), prediction.data.numpy(), c='red', lw=5)


save()
restore_net()
restore_net_params()

plt.show()