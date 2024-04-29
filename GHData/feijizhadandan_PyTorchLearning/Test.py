import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# 激活函数使用 relu(), 不用 sigmoid()了
import torch.nn.functional as functional
import torch.optim as optim


transform = transforms.Compose([
    # 能将数据集中的图像转换成 1*28*28 的三维张量
    transforms.ToTensor(),
    # 将数据从 0~255 压缩到 0~1 之间的 0-1分布 (训练效果好)
    transforms.Normalize((0.1307, ), (0.3081, ))
])
'''
    下载数据集
'''
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

print(len(train_loader.dataset))
print(len(train_loader))
