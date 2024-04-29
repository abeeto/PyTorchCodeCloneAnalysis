# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:54:12 2019

@author: gx@NJUESE
"""

import torch
import torchvision
import numpy as np
'''
torchvision包，它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
'''
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
start = time.time()
mnist_train = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
'''
指定了参数transform = transforms.ToTensor()使所有数据转换为Tensor，如果不进行转换则返回的是PIL图片。
transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组
转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor
'''

#feature, label = mnist_train[9]
#print(feature.shape, label)  # Channel x Height x Width

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

'''    
X, y = [], []
for i in range(9):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
'''

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

'''
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
'''

input_num = 784 #图像大小为28*28
output_num = 10 #10分类问题
w = torch.tensor(np.random.normal(0, 0.01, (input_num, output_num)), dtype=torch.float)
b = torch.zeros(output_num, dtype=torch.float)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(X):
    X_exp = X.exp()
    sigma = X_exp.sum(dim=1, keepdim=True) #
    return X_exp / sigma
'''
给定一个Tensor矩阵X。我们可以只对其中同一列（dim=0）或同一行（dim=1）的元素求和，
并在结果中保留行和列这两个维度（keepdim=True）。
'''

def net(X):
    return softmax(torch.mm(X.view(-1,input_num),w)+b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def sgd(params, lr, batch_szie):
    for param in params:
        param.data -= lr*param.grad / batch_size
        

num_epochs, lr = 40, 0.08


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    l_dis = []
    test_acc_d = []
    train_acc_d = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        l_dis.append(train_l_sum / n)
        test_acc_d.append(train_acc_sum / n)
        train_acc_d.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    return l_dis, test_acc_d, train_acc_d

l_dis,test_acc,train_acc = train_ch3(net, train_iter, test_iter, 
                                     cross_entropy, num_epochs, batch_size, [w, b], lr)

X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(1,num_epochs+1), l_dis, label='loss')
ax1.set_ylabel('Loss')
ax1.set_title("Training Result")
#plt.legend()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(range(1,num_epochs+1), test_acc,'g', label='test_accuracy')
ax2.plot(range(1,num_epochs+1), train_acc,'r', label='train_accuracy')
ax2.set_ylabel('Accuracy')
fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
'''
plt.title("Training Result")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(range(1,num_epochs+1), l_dis)
'''
print('softmax_reg PyTorch CPU ver %.2f sec' % (time.time() - start))
