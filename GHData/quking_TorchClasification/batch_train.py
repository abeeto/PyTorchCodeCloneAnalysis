import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torchvision


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWN = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),  #
    download=DOWN,
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.0
"""
test_data.test_data中的shape为[10000, 28, 28]代表1w张图像，都是28x28，当时并未表明channels,因此在unsqueeze在1方向想加一个维度，
则shape变为[10000, 1, 28, 28]，然后转化为tensor的float32类型，取1w张中的2000张，并且将其图片进行归一化处理，避免图像几何变换的影响
"""
# 标签取前2000
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #前面都是规定结构
        #第一个卷积层，这里使用快速搭建发搭建网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#灰度图，channel为一
                out_channels=16,#输出channels自己设定
                kernel_size=3,#卷积核大小
                stride=1,#步长
                padding=1#padding=（kernel_size-stride）/2   往下取整
            ),
            nn.ReLU(),#激活函数，线性转意识到非线性空间
            nn.MaxPool2d(kernel_size=2)#池化操作，降维，取其2x2窗口最大值代表此窗口，因此宽、高减半，channel不变
        )
        #此时shape为[16, 14, 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #此时shape为[32, 7, 7]
        #定义全连接层，十分类，并且全连接接受两个参数，因此为[32*7*7, 10]
        self.prediction = nn.Linear(32*7*7, 10)
        #前向传播过程

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*7*7)
        output = self.prediction(x)
        return output


# 创建网络
cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), LR)
loss_func = nn.CrossEntropyLoss()
# 训练阶段
for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(train_loader):
        # model只接受Variable的数据，因此需要转化
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        # 将b_x输入到model得到返回值
        output = cnn(b_x)
        # 计算误差
        loss = loss_func(output, b_y)
        # 将梯度变为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
        # 打印操作，用测试集检验是否预测准确
        if step%50 == 0:
            test_output = cnn(test_x)
            # squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]
            pre_y = torch.max(test_output, 1)[1].data.squeeze()
            # 总预测对的数除总数就是对的概率
            accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" %accuracy)

