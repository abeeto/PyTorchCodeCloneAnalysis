import math
import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


class VGG16(nn.Module):
    def __init__(self, num_class=10):
        super(VGG16, self).__init__()
        self.num_class = num_class

        # 14层卷积
        self.Conv = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # conv2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # conv4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # conv6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # conv7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # conv9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # conv10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # conv12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # conv13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # # conv14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 两层全连接
        self.FC = nn.Sequential(
            # fc15
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),

            # fc16
            nn.Linear(256, self.num_class),
            # nn.Softmax(dim=1)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, img):
        img = self.Conv(img)
        img = img.view(img.size(0), -1)
        outputs = self.FC(img)
        return outputs


class Cutout(object):
    """图片处理"""

    def __init__(self, hole_size):
        # 正方形马赛克的边长，像素为单位
        self.hole_size = hole_size

    def __call__(self, img):
        return cutout(img, self.hole_size)


def cutout(img, hole_size):
    y = np.random.randint(32)
    x = np.random.randint(32)

    # 以(x,y)为中心，裁取hole_size*hole_size大小的区域
    half_size = hole_size // 2
    x1 = np.clip(x - half_size, 0, 32)
    x2 = np.clip(x + half_size, 0, 32)
    y1 = np.clip(y - half_size, 0, 32)
    y2 = np.clip(y + half_size, 0, 32)

    # 转换图片为array形式
    imgnp = np.array(img)

    # 将此区域马赛克，即将值设为0
    imgnp[y1:y2, x1:x2] = 0

    # 将图片转化回RGB形式
    img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
    return img


def load_data(batch_size):
    """加载数据集"""
    train_transform = transforms.Compose([
        # 对原始32*32图像四周各填充4个0像素（变为40*40），然后随机裁剪成32*32
        transforms.RandomCrop(32, padding=4),

        # 随机马赛克，大小为6*6
        Cutout(6),

        # 按0.5的概率水平翻转图片
        transforms.RandomHorizontalFlip(0.5),

        # 随机改变图像的亮度，对比度、饱和度等
        transforms.ColorJitter(0.4, 0.4, 0.4),

        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_dataset = datasets.CIFAR10(root="./", train=True, transform=train_transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    test_dataset = datasets.CIFAR10(root="./", train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_dataset, train_loader, test_dataset, test_loader


def train_model(model, train_loader, optimizer, criterion, scheduler, train_dataset, epoch, gpu_is_available):
    """模型训练"""
    model.train()

    # 模型损失与正确率
    L = 0
    acc = 0

    for i, train_data in enumerate(train_loader):
        img_train, label_train = train_data
        if gpu_is_available:
            img_train = img_train.cuda()
            label_train = label_train.cuda()

        # 参数梯度置零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = model(img_train)
        loss = criterion(outputs, label_train)
        loss.backward()
        optimizer.step()

        L += loss.item()
        # 最大值所在的位置索引
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == label_train).sum()
        acc += num_correct.item()

    # 学习率衰减
    scheduler.step()
    print("=====================epoch{}=====================".format(epoch + 1))
    # 训练集的正确率
    acc = acc / len(train_dataset)
    # 为了数值美观和观察方便，扩大1000倍
    L = L / len(train_dataset) * 1000
    print(f'训练集损失为:{L}，训练集正确率:{acc * 100:.2f}%')
    return L


def test_model(model, test_loader, test_dataset, gpu_is_available):
    """模型测试"""
    model.eval()

    # 模型正确率
    acc = 0

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            img_test, label_test = test_data

            if gpu_is_available:
                img_test = img_test.cuda()
                label_test = label_test.cuda()

            outputs = model(img_test)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == label_test).sum()
            acc += num_correct.item()

        # 测试集的正确率
        acc = acc / len(test_dataset)
        print(f'测试集正确率: {acc * 100:.2f}%')
        return acc


def cost_visualization(epoches, LOSS):
    """损失函数可视化"""
    plt.figure(figsize=(8, 4), dpi=100)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.xlabel("迭代次数", fontdict={'size': 14})
    plt.ylabel("损失", fontdict={'size': 14})
    plt.title("损失函数图")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13)
    plt.plot(range(1, epoches + 1), LOSS, linewidth=2.5, label="cost")
    plt.legend(loc=0)
    plt.savefig("./picture/cost_picture.png")
    plt.show()


def acc_visualization(epoches, ACC):
    """准确率可视化"""
    plt.figure(figsize=(8, 4), dpi=100)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.xlabel("迭代次数", fontdict={'size': 14})
    plt.ylabel("准确率", fontdict={'size': 14})
    plt.title("准确率")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13)

    def to_percent(temp, position):
        return '%.0f' % (100 * temp) + '%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.plot(range(1, epoches + 1), ACC, linewidth=2.5, label="accuracy")
    plt.legend(loc=0)
    plt.savefig("./picture/acc_picture.png")
    plt.show()
