# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 16:51

import torch
import torchvision
from numpy import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from model_1 import *

train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集长度为：{}".format(train_data_size))
print("测试集长度为：{}".format(test_data_size))

# 定义使用设备
device = torch.device("cpu")
# 加载数据集
train_data_loader = DataLoader(train_data, 64)
test_data_loader = DataLoader(test_data, 64)

# 创建模型
model = Model()
model.to(device)

# 创建损失函数
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# 创建优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

# tensorboard
writer = SummaryWriter("./logs")

# 计时
start_time = time.time()
for i in range(epoch):
    print("———— 第{}轮开始 ————".format(i+1))

    # 训练
    model.train()
    for data in train_data_loader:
        img, target = data
        if torch.cuda.is_available():
            img = img.to(device)
            target = target.to(device)
        output_img = model(img)
        loss = loss_func(output_img, target)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("time is: ", (end_time - start_time))
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("tran_loss", loss.item(), total_train_step)

    # 验证 在训练一段时间之后，利用测试集进行检验，with torch.no_grad() —— 当前梯度不改变
    # (相当于是每个epoch之后进行一次验证)
    model.eval()
    test_loss = []
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            img, target = data
            if torch.cuda.is_available():
                img = img.to(device)
                target = target.to(device)
            output_img = model(img)
            loss = loss_func(output_img, target)
            test_loss.append(loss.item())

            # 评估结果（计算正确率）
            accuracy = (output_img.argmax(1) == target).sum()       # 所有预测正确的个数
            total_accuracy += accuracy
    mean_loss = sum(test_loss)/len(test_loss)
    print("测试平均Loss：{}".format(mean_loss))
    print("测试正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", mean_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    # torch.save(model, "./pth/mymodel_{}.pth".format(i))
    torch.save(model.state_dict(), "mymodel_{}.pth".format(i))

    writer.close()
