# 导入pytorch框架
import torch
import torchvision
# 导入transforms方法集合
import torchvision.transforms as transforms
# 导入系统变量
import os
import sys

# 表示当前所处的目录位置，方便获取数据地址
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]


def data_process():
    # Transform方法
    transform_train = transforms.Compose([
        # 对图像外围进行补0操作，然后将其随即裁剪为32*32
        transforms.RandomCrop(32, padding=4),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为Tensor张量
        transforms.ToTensor(),
        # 归一化，均值为(0.4914, 0.4822, 0.4465)，方差为(0.2023, 0.1994, 0.2010)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        # 转换为Tensor张量
        transforms.ToTensor(),
        # 归一化，均值为(0.4914, 0.4822, 0.4465)，方差为(0.2023, 0.1994, 0.2010)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dataset类
    # 这里不是没有getitem方法，而是这个数据集太常用了，框架把他封装好了，我们只需要自己写transform就行了
    trainset = torchvision.datasets.CIFAR10(
        root=path + '/data', train=True, download=True, transform=transform_train)  # 我们把transform放到了Dataset类里面
    testset = torchvision.datasets.CIFAR10(
        root=path + '/data', train=False, download=True, transform=transform_test)

    # DataLoader类
    # batch size=10， 打乱顺序，多线程读数据
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size='code1', shuffle='code2', num_workers='code3')  # 我们把Dataset放到了DataLoader类里面
    # batch size=10， 不打乱顺序，多线程读数据
    testloader = torch.utils.data.DataLoader(
        testset, batch_size='code4', shuffle='code5', num_workers='code6')

    return trainloader, testloader


