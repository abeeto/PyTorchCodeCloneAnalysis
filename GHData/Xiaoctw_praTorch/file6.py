from numpy import *
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as fun


# 参考定义一个神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)  # 核的大小为5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 头一步相当于恢复到原有形状？？？，放大了16倍？？？
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = fun.relu(self.conv1(x))
        print(x.shape)
        x = fun.max_pool2d(x, (2, 2))  # 最大池化层
        print(x.shape)
        x = fun.relu(self.conv2(x))
        print(x.shape)
        x = fun.max_pool2d(x, 2)
        print(x.shape)
        # 将多维数据转化为2维数据
        x = x.view(-1, self.num_flat_features(x))
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 这个函数相当于计算除了第一维之外所有维度的乘积
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
# params=list(net.parameters())
# print(params)
in_put = torch.randn(1, 1, 35, 35)  # 这里的第一维相当于是数量，是多少都没有关系，但是第二维代表的是通道数，必须和网络一致
out = net(in_put)  # 这里得到的out为一个向量，在反向传播是需要加上参数
print(out.shape)
# print(out)
# net.zero_grad()
# out.backward(torch.randn(1,10))
learning_rate = 1e-6
# for f in net.parameters():
#     #f.data.sub_(f.grad.data*learning_rate)
#     f.data-=f.grad.data*learning_rate
optimizer = optim.SGD(net.parameters(), learning_rate)
optimizer.zero_grad()
target = torch.randn(out.size())
criterion = nn.MSELoss(reduction='sum')  # 代替size_average=False
loss = criterion(out, target)
loss.backward()
optimizer.step()
