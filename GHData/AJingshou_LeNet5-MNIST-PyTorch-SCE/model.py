import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
#        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
'''
 
''' 
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        # 包含一个卷积层和池化层，分别对应LeNet5中的C1和S2，
        # 卷积层的输入通道为1，输出通道为6，设置卷积核大小5x5，步长为1
        # 池化层的kernel大小为2x2
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # 包含一个卷积层和池化层，分别对应LeNet5中的C3和S4，
        # 卷积层的输入通道为6，输出通道为16，设置卷积核大小5x5，步长为1
        # 池化层的kernel大小为2x2
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # 对应LeNet5中C5卷积层，由于它跟全连接层类似，所以这里使用了nn.Linear模块
        # 卷积层的输入通特征为4x4x16，输出特征为120x1
        self._fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        # 对应LeNet5中的F6，输入是120维向量，输出是84维向量
        self._fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        # 对应LeNet5中的输出层，输入是84维向量，输出是10维向量
        self._fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )
 
    def forward(self, input):
        # 前向传播
        # MNIST DataSet image's format is 28x28x1
        # [28,28,1]--->[24,24,6]--->[12,12,6]
        conv1_output = self._conv1(input)
        # [12,12,6]--->[8,8,,16]--->[4,4,16]
        conv2_output = self._conv2(conv1_output)
        # 将[n,4,4,16]维度转化为[n,4*4*16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)
        # [n,256]--->[n,120]
        fc1_output = self._fc1(conv2_output)
        # [n,120]-->[n,84]
        fc2_output = self._fc2(fc1_output)
        # [n,84]-->[n,10]
        fc3_output = self._fc3(fc2_output)
        return fc3_output
    
    def num_flat_features(self, x):
        # Get the number of features in a batch of tensors x
        size = x.size()[1:]
        return np.prod(size)
    '''
class Model(nn.Module):

    # network structure
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)