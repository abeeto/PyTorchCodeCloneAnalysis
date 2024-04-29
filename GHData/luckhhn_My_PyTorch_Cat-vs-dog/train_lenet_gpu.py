# -*- coding: utf-8 -*-

'''
@Time    : 2020/6/11 23:54
@Author  : HHNa
@FileName: train_lenet_gpu.py
@Software: PyCharm
 
'''

# -*- coding: utf-8 -*-
"""
# @file name  : train_lenet.py
# @author     : tingsongyu
# @date       : 2019-09-07 10:08:00
# @brief      : 人民币分类模型训练
"""
"""
# @file name  : train_lenet_gpu.py
# @modified by: greebear
# @date       : 2019-10-26 13:25:00
# @brief      : 猫狗分类模型训练
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from model.lenet_week import LeNet, MyNet
from tools.DogCat_dataset import DogCatDataset
from tools.common_tools import transform_invert

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()  # 设置随机种子

# 参数设置
MAX_EPOCH = 40
BATCH_SIZE = 222#512 224内存满
LR = 0.001
log_interval = 10
val_interval = 1

# import torch
# a = torch.cuda.is_available()
# print(a)

# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print(device)
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3,3).cuda())


if __name__ == '__main__':

    # ============================ step 1/5 数据 ============================

    split_dir = os.path.join(".", "data", "cad_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = DogCatDataset(data_dir=train_dir, transform=train_transform)
    valid_data = DogCatDataset(data_dir=valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=16)

    # ============================ step 2/5 模型 ============================

    net = MyNet(classes=2)
    net.initialize_weights()
    print(net.to("cuda"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 优化器 ============================
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-1)    # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)    #设置学习率下降策略

    from torchsummary import summary
    summary(net, input_size=(3, 112, 112))

    valid_acc = list()
    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):
            # forward
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss.item())
                valid_acc.append(correct_val / total_val)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} 【Acc:{:.2%}】".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val, correct_val / total_val))

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()
    torch.save(net, "./1")
