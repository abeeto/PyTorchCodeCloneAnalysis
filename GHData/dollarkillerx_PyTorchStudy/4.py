# 数据归一
    # 图像数据像素值一般在[0-255]之间
    # 在训练网络时，我们经常把输入数据值变成[0-1]或者[-1,-1]之间
# PyTorch库
    # 数据加载
        # torchvision.dataset
            # 知名公共数据集可用torchvision.dataset.数据集名称加载
                # 例如: torchvision.datasets.CIFAR10 加载CIFAR10数据集
            # 私人数据集可用torchvision.dataset.ImageFolder和
            # torch.utils.data.DataLoader加载
    # 数据归一
        # torchvision.transforms
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose( # 定义归一方法
    [
        transforms.ToTensor(), # 加载图片转变为tensor格式
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # 标准化归一  parmas1: 图片三个rgb通道的平均值,  params2：图片的标准偏差 
        # transforms.Flip(), 如果输入数据不够  可用改变当前输入 比如 对当前数据进行反转 等 
    ]
)

# 训练数据集
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
                                        # 数据存储地址   是否是训练  是否下载    归一方法 
    
    # 定义加载方法
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
                                        # 训练数据   每次加载个数  是否打乱每次训练的数据  同时有几个线程在加载
    
# 测试数据集
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

    # 定义加载方法
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

%matplotlip inline

def imshow(img):
    # 输入数据: torch.tensor [c,h,w]
    img = img / 2 + 0.5  # 反归一
    nping = img.numpy()
    
    