# -*- encoding: utf-8 -*-
"""
@File    :   train.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/29 20:23   thgpddl      1.0         None
"""
import torch
import torch.nn as nn
from torch import optim
import os

from model import LeNet
from data import getLoader


# 超参数
epoch = 10

# 数据加载器
data_train_loader, data_test_loader = getLoader()

# 模型实例化
model = LeNet()
model.train()

# 优化器配置
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 训练
train_loss, correct, total = 0, 0, 0
for i in range(epoch):
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        optimizer.zero_grad()  # 每次需要将梯度缓存置0，防止影响下次更新
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 根据loss计算梯度(但是不更新)，梯度值保存在每个相关parameter的属性中
        optimizer.step()  # 根据梯度更新网络参数

        # var.item()：得到var的值,数据类型为元素的数据类型(不一定是原来的数据类型)
        train_loss += loss.item()  # 这里是将训练过程中所有的loss相加，即Σloss

        # torch.max(tensor,dim)：dim是max函数索引的维度0，0是每列的最大值，1是每行的最大值
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, predicted = outputs.max(1)

        total += targets.size(0)  # 获取batch_size
        correct += predicted.eq(targets).sum().item()  # 获取预测正确的个数，correct是总正确个数，即Σcorrect

        if batch_idx % 10 == 0:
            print("epoch：%d  " % i, batch_idx, len(data_train_loader),
                  "Loss: %.3f | Acc: %.3f%%(%d/%d)" % (
                  train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    save_info = {
        "item_epoch_num": epoch,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
    }
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    torch.save(save_info, "outputs/epoch_{}_acc_{}%.pth".format(i, str(100. * correct / total)[0:5]))
