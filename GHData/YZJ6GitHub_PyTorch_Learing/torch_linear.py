import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

#pytorch处理的数据是二维的，因此需要利用unsqueeze将一维数据转换为二维
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x),Variable(y)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module): # 继承 torch 的 Module
    def __init__(self,n_feature,n_hidden,n_output):  # 继承 __init__ 功能
        super(Net,self).__init__()
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,n_hidden) # 隐藏层线性输出
        self.prediect = torch.nn.Linear(n_hidden,n_output) # 输出层线性输出

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.prediect(x)
        return x
net = Net(n_feature = 1,n_hidden = 10,n_output= 1)
print(net)

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)# 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for i in range(100):

    prediction = net(x) # 喂给 net 训练数据 x, 输出预测值
    loss = loss_func(prediction,y)
    optimizer.zero_grad()    # 计算两者的误差
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    if i %5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)




















