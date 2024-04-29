import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision

import torch.optim as optim
#BATCH_SIZE = 64
import torch.nn.functional as F
import time


import torch.nn as nn

from collections import OrderedDict

from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    growth = 2

    def __init__(self, inputs, cardinality=32, block_width=4, stride=1):
        super(Block, self).__init__()
        outs = cardinality*block_width
        self.left = nn.Sequential(
            nn.Conv2d(inputs, outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, outs, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, self.growth * outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.growth * outs)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inputs != self.growth * outs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, self.growth * outs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.growth * outs)
            )

    def forward(self, inputs):
        network = self.left(inputs)
        network += self.shortcut(inputs)
        out = F.relu(network)
        return out


class ResnextNet(nn.Module):
    def __init__(self, layers, cardinality, block_width):
        super(ResnextNet, self).__init__()
        self.inputs = 64
        self.cardinality = cardinality
        self.block_width = block_width

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv12 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=1, stride=1, bias=False)
        )
        self.conv1 = F.relu(self.conv11 +self.conv12)
        self.conv2 = self._block(layers=layers[0], stride=1)
        self.conv3 = self._block(layers=layers[1], stride=2)
        self.conv4 = self._block(layers=layers[2], stride=2)
        self.conv5 = self._block(layers=layers[3], stride=2)

        self.linear = nn.Linear(8 * cardinality * block_width, 10)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        print(network.shape)
        network = F.avg_pool2d(network, kernel_size=network.shape[2]//2)
        print(network.shape)
        network = network.view(network.size(0), -1)
        out = self.linear(network)

        return out, network

    def _block(self, layers, stride):
        strides = [stride] + [1] * (layers - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(Block(self.inputs, self.cardinality, self.block_width, stride))
            self.inputs = self.block_width * self.cardinality * Block.growth
        return nn.Sequential(*layers)


def ResNext50_32x4d():
    return ResnextNet(layers=[3, 4, 6, 3], cardinality=32, block_width=4)


def ResNext50_4x32d():
    return ResnextNet(layers=[3, 4, 6, 3], cardinality=4, block_width=32)


def ResNext101_32x4d():
    return ResnextNet(layers=[3, 4, 23, 3], cardinality=32, block_width=4)


def ResNext101_64x4d():
    return ResnextNet(layers=[3, 4, 23, 3], cardinality=64, block_width=4)


transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.Resize((32,32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),

    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

# transform  = T.Compose([
#
#          T.RandomRotation(5),
#          T.Resize(32),
#          T.ToTensor()
# ])
datatrain = ImageFolder('E:/sdxxsjj/traintry/',transform=transform_train)
datatest=ImageFolder('E:/sdxxsjj/testtry/',transform=transform_test)

print(datatrain[0][0].size())

datatrain1=datatrain
datatest1=datatest


trainloader = torch.utils.data.DataLoader(datatrain1, batch_size=128,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(datatest1,batch_size=128,shuffle=False,num_workers=0)

net = ResNext50_4x32d()

# #损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
# #优化器这里用ADAM，一阶距和二阶距的指数衰减率
optimizer = optim.Adam(net.parameters(),lr=0.1,betas=(0.9,0.99),eps=1e-06, weight_decay=0.002)
#选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载网络
net.to(device)

##断点续训
# checkpoint = torch.load('modelpara.pth')
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# initepoch = checkpoint['epoch']+1
# net.eval()
#
#
print("开始训练!")
num_epochs = 1#训练次数
#
i=1
for epoch in range(num_epochs):
    running_loss = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        #print(i)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, aus = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()#梯度初始化为零
        loss.backward()#反向传播
        optimizer.step()#更新所有参数
        print(i, loss.item())
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, 'modelpara.pth')
    print('经过%d个epoch后,损失为:%.4f'%(epoch+1, loss.item()))
    i=i+1
    str1 = 'zuizhong' + str(i) + '.pkl'
    torch.save(net, str1)
print("结束训练")
#保存训练模型


#加载训练模型
net = torch.load(str1)
#开始识别
with torch.no_grad():
    #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out,ous = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('测试集图片的准确率是:{}%'.format(100 * correct / total)) #输出识别准确率
