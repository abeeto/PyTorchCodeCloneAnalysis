import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


# 自己定义一个神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3->6,5*5大小
        self.bn1 = nn.BatchNorm2d(6)  # 这里应该是通道数
        self.pool = nn.MaxPool2d(2, 2)  # 定义最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(fun.relu(self.bn1(self.conv1(x))))
        x = self.pool(fun.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)  # 相当于展平
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
定义一种转化方式
进行一步转化，转换到0-1之间
(（0,1）-0.5)/0.5=(-1,1)
``input[channel] = (input[channel] - mean[channel]) / std[channel]``
"""
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainSet = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)
testSet = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testLoader = torch.utils.data.DataLoader(testSet,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2
                                         )
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse'
                                                                 'ship', 'truck']


# 定义显示图片的方式
def imshow(img):
    img = img / 2 + 0.5  # 相当于之前的逆过程
    npimg = np.array(img)  # 转化为numpy的格式，首先转换数据类型，再交换轴
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dataIter = iter(trainLoader)
images, labels = dataIter.__next__()
print(labels)
# 结果为4,3,32,32
print(images.shape)
# 结果为3,32,32
print(images[0].shape)
print(labels.shape)

# imshow(torchvision.utils.make_grid(images))

net = Net()
net = net.to(device)  # 设置好设备
# 这个不可以用！！！criterion=nn.MSELoss(reduction='sum')
# 分类交叉熵
criterion = nn.CrossEntropyLoss(reduction='sum')
# optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
optimizer=torch.optim.Adam(net.parameters(),lr=3e-5)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data  # 每组数据有四个，还有标签
        # 这一步就相当于适配设备
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)  # 这个是分批次进行训练，没有把所有数据全部塞到网络中
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0
print("训练完成")
outputs = net(images)  # 对图像类别进行预测
print(outputs.shape)
_, predicted = torch.max(outputs, 1)  # 返回的是值和其对应的下表,在维度1上进行比较
print("Predicted: ", " ".join("%5s" % classes[predicted[i]] for i in range(4)))  # 一共是4张图片，在上面进行预测

correct, total = 0, 0
with torch.no_grad():  # 这里意思就是不需要计算反向梯度，因为模型已经训练好了
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()  # 获取数据大小，不加也可以
print("测试集上准确度:{}".format(correct / total))
