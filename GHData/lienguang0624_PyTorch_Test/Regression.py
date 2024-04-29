import  torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义输入与输出
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x),Variable(y)

# 定义神经网络模型
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return  x

# 创建神经网络
net = Net(1,10,1)

# 定义优化方式（随机梯度下降）
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)

# 定义损失函数计算方式
loss_func = torch.nn.MSELoss()

# 可视化
plt.figure()
plt.ion()
plt.show()

# 训练
for i in range(1000):
    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()# 梯度设为零
    loss.backward()# 反向传递
    optimizer.step()# 优化梯度
    if i % 5 == 0:
        print('\nepoc:',i)
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw = 5)
        plt.pause(0.1)

plt.ioff()
plt.show()
