#Lenet5 example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()# 26.定义①的卷积层，输入为32x32的图像，卷积核大小5x5卷积核种类6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 27.定义③的卷积层，输入为前一层6个特征，卷积核大小5x5，卷积核种类16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 28.定义⑤的全链接层，输入为16*5*5，输出为120
        self.fc1 = nn.Linear(16*5*5, 120)  # 6*6 from image dimension# 29.定义⑥的全连接层，输入为120，输出为84
        self.fc2 = nn.Linear(120, 84)# 30.定义⑥的全连接层，输入为84，输出为10
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 31.完成input-S2，先卷积+relu，再2x2下采样
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))# 32.完成S2-S4，先卷积+relu，再2x2下采样
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #卷积核方形时，可以只写一个
        # 33.将特征向量扁平成列向量
        x = x.view(-1, 16*5*5)
        # 34.使用fc1+relu
        x = F.relu(self.fc1(x))
        # 35.使用fc2+relu
        x = F.relu(self.fc2(x))
        # 36.使用fc3
        x =self.fc3(x)
        return x
net = Net()
print(net)
params =list(net.parameters())
#print(params)
print(len(params))
print(params[0].size())
input= torch.randn(1,3, 32, 32)
out = net(input)
criterion=nn.MSELoss()
target=torch.randn([1,10])
output=net(input)
loss=criterion(output,target)
print(loss)
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
# 更新权重
optimizer.step()
x=range(10)
x=iter(x)
x=torch.rand(2,3)
print(x.shape[0])
ch=x.char()
print(ch)