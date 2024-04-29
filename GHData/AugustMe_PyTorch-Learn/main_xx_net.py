# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:36:59 2019

@author: ZQQ
"""

import torch
from torch import nn, optim
from torch.autograd import Variable  # 导出Variable
from torch.utils.data import DataLoader #导入DataLoader，在torch中需要dataloader进行迭代
from torchvision import datasets, transforms # 导入pytorch内置的数据集

from models import *  # 从models文件夹导入所有网络
from models import lenet

# 定义一些超参数
batch_size = 64
learning_rate = 0.02 # 学习率
num_epoches = 1

# 数据预处理，组合起来
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]) # 黑白图片这么处理，彩色图片是三通道的

# 数据集的下载器
train_dataset = datasets.MNIST(root='./MNIST_data', 
                               train=True,
                               transform=data_tf,
                               download=False) # 为True需要先下载，然后处理；为False使用下载完成的

test_dataset = datasets.MNIST(root='./MNIST_data', 
                              train=False, 
                              transform=data_tf
                              )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #训练集，shuffle为True时打乱数据，让数据更有选择随机性
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #对测试集进行迭代器编号

# 60000张训练集，batch_size为50 => 60000/50=1200,即每次模型输入50个数据，要进行1200次
# print(len(train_loader))

# 选择模型
model = lenet.LeNet()
#print(model) # 查看模型架构
criterion = nn.CrossEntropyLoss() # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 定义优化器

#'''训练之前定义一些空列表来添加训练得到的一些值 '''
train_losses = []
train_acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epoches):
    # 每次epoch训练
    train_loss = 0
    train_acc = 0
    model.train()    
    # 训练模型
    #for data in train_loader: # 循环遍历len(train_loader)次,每个data,batch_size张图，batch_size个真实标签
    for img, label in train_loader:
        #img, label = data
        img = Variable(img)
        #img = img.view(img.size(0), -1)  # 如果在定义的网络中没有，则此处需要将数据拉平
        label = Variable(label)
        #print(label)
        ## 前向传播
        out = model(img)
        #out = model(img)[0] #  output
        #print(out)
        loss = criterion(out, label)
        #print(loss)
        #print_loss = loss.data.item()
        ## 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## 记录误差 
        train_loss += loss.item() #我们需要的是"所有数据"前后传播一次，也就是len(train_loader)次的loss的和
        #print(train_loss)
        ## 计算分类准确率
        _,pred = out.max(1) # pred为一次batch数据的预测标签，挑选出输出时值最大的位置
        #print(pred)
        num_correct = (pred == label).sum().item() #记录正确的个数
        #print(num_correct)
        acc = num_correct / img.shape[0] #计算精确率，img.shape[0]：此次批数据大小
        train_acc += acc
        #print(train_acc)
    
    # 一个epoch后，记录数据    
    train_losses.append(train_loss / len(train_loader)) # 得到的平均训练误差添加到列表中
    train_acces.append(train_acc / len(train_loader))   # 得到的平均训练精度添加到列表中
    #print(train_losses)

    # 模型评估
    model.eval() #将模型改为预测模式
    eval_loss = 0
    eval_acc = 0
    #for data in test_loader: # 循环遍历len(test_loader)次,每个data,batch_size张图，batch_size个真实标签
        #img, label = data
    for img, label in test_loader:       
        img = Variable(img)
        label = Variable(label)
        # 前向传播，测试不需要反向传播
        out = model(img)
        #print(out)
        loss = criterion(out, label)
        #print(loss)
        #eval_loss += loss.data.item()
        eval_loss += loss.item()
        #print(eval_loss)
        # 记录准确率
        _,pred = out.max(1)
        #_, pred = torch.max(out, 1)
        #num_correct = (pred == label).sum()
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0] # img.shape: 就是batch_size个数
        eval_acc += acc
    # 完成一次遍历 
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    
    # 输出
    print('Epoch:',epoch,
          '|Train Loss: %.6f' % (train_loss / len(train_loader)),
          '|Train Acc: %.6f' % (train_acc / len(train_loader)), 
          '|Eval Loss: %.6f' % (eval_loss / len(test_loader)),
          '|Eval Acc: %.6f' % (eval_acc / len(test_loader))
          )

import numpy as np
print('平均准确率：{:.4f}'.format( np.array(eval_acces).sum() / len(eval_acces) )) 










