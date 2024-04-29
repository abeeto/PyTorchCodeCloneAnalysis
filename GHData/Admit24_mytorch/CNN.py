import torch
import torch.nn as nn
import torch.nn.functional as F


'''
CNN计算

(H - k +2 * P) / S + 1
(W - k +2 * P) / S + 1

LetNet-5 
input: 32*32*3

out_conv1 = (32-5)+1 = 28 
max_pool1 = 28 / 2 = 14
out_conv2 = (14 - 5) + 1 = 10
max_pool2 = 10 / 2 = 5
'''

'''

定义一个神经网络

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #  conv1层，输入的灰度图，所以 in_channels=1, out_channels=6 说明使用了6个滤波器/卷积核，
        # kernel_size=5卷积核大小5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # conv2层， 输入通道in_channels 要等于上一层的 out_channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # an affine operarion: y = Wx + b
        # 全连接层fc1,因为32x32图像输入到fc1层时候，feature map为： 5x5x16
        # 因此，全连接层的输入特征维度为16*5*5，  因为上一层conv2的out_channels=16
        # out_features=84,输出维度为84，代表该层为84个神经元
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特征图转换为一个１维的向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
