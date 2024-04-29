import torch
import torchvision # 服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import torch.utils.data as Data
from d2lzh_pytorch.utils import *

# Fashion-MNIST是一个10类服饰分类数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]  # 通过下标来访问任意一个样本:
print(feature.shape, label)  # Channel x Height x Width  第一维是通道数，因为数据集中是灰度图像，所以通道数为1。后面两维分别是图像的高和宽。

# chakan看训练数据集中前10个样本的图像内容和文本标签
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4  # 通过参数num_workers来设置4个进程读取数据
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间。
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
