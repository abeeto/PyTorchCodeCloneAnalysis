import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision
import torch.nn as nn
import torch.optim as optim
#BATCH_SIZE = 64
import torch.nn.functional as F
import time


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=500):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),

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
datatrain = ImageFolder('E:/sdxxsjj/train/',transform=transform_train)
datatest=ImageFolder('E:/sdxxsjj/test/',transform=transform_test)

print(datatrain[0][0].size())
# print(datatrain[1][0])
# print(datatest.class_to_idx)
# to_img = T.ToPILImage()
# a=to_img(datatrain[1][0])
# plt.imshow(a)
#
# print(datatrain[1707][1])
# plt.show()

datatrain1=datatrain
datatest1=datatest


trainloader = torch.utils.data.DataLoader(datatrain1, batch_size=32,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(datatest1,batch_size=32,shuffle=False,num_workers=0)
#print(trainloader)
net = ResNet152()
# #损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
# #优化器这里用ADAM，一阶距和二阶距的指数衰减率
optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=(0.9,0.99))
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

for epoch in range(num_epochs):
    running_loss = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        #print(i)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)



        loss = criterion(outputs, labels)
        optimizer.zero_grad()#梯度初始化为零
        loss.backward()#反向传播
        optimizer.step()#更新所有参数
        print(loss.item())
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, 'modelpara.pth')

    print('经过%d个epoch后,损失为:%.4f'%(epoch+1, loss.item()))
print("结束训练")
#保存训练模型
torch.save(net, 'zuizhong.pkl')
#加载训练模型
net = torch.load('zuizhong10lei.pkl')
#开始识别
with torch.no_grad():
    #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('测试集图片的准确率是:{}%'.format(100 * correct / total)) #输出识别准确率
