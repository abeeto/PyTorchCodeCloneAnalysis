#1、学习如何用PyTorch做个回归：regression
#2、学习两种保存模型的方法
#3、学习DataLoader

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-1,1,100)
#torch.squeeze()是用来对数据的维度进行压缩，去掉维数为1的维度
#squeeze(a)就是将a中所有为1的维度删掉，不为1的没有影响
#a.squeeze(N) 就是去掉a中指定维数为1的维度 还有一种形式就是b = torch.squeeze(a,N)
#torch.unsqueeze(x,dim = 1) 用来增加维度
x = torch.unsqueeze(x,dim = 1) #增加一个维度
y = x.pow(2) + 0.2 * torch.randn(x.size())
x,y = Variable(x),Variable(y)

#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)


    def forward(self,x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1,10,1)
print(net)  #打印一下搭建的神经网络的结构

plt.ion()  #变成实时打印的过程
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_fn = nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_fn(prediction,y)  #定义损失函数
    print(t,loss.item())

    optimizer.zero_grad()  #清空梯度
    loss.backward()  #损失函数求导
    optimizer.step() #使用SGD更新参数

    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

#下面介绍两种保存模型的方法
#第一种是保存整个模型：
torch.save(net,'net.pkl')
#如何加载？
net2 = torch.load('net.pkl')

#第二种是仅仅保存模型的参数：
torch.save(net.state_dict(),'net_para.pkl')
#这种方式又如何加载呢？
#首先需要建立一个与net的结构一模一样的网络
net3 = Net(1,10,1)
net3.load_state_dict(torch.load('net_para.pkl'))  #第二种方法的效率高，推荐

import torch.utils.data as Data
torch_data = Data.TensorDataset(data_tensor = x,target_tensor = y)
loader = Data.DataLoader(
    dataset = torch_data,
    batch_size = 10,
    shuffle = True,
    num_workers = 2,
)


