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
BATCH_SIZE = 222  # 512 224内存满
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
    # ============================ inference ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    net = torch.load("./1.pth")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(BASE_DIR, "test_data")

    test_data = DogCatDataset(data_dir=test_dir, transform=valid_transform)
    valid_loader = DataLoader(dataset=test_data, batch_size=1)
    # net = MyNet(classes=2)
    for i, data in enumerate(valid_loader):
        # forward
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        outputs = net(inputs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        _, predicted = torch.max(outputs.data.cpu(), 1)

        rmb = 'dog' if predicted.numpy()[0] == 0 else 'cat'
        print("模型获得动物是{}".format(rmb))

        img_tensor = inputs[0, ...]  # c H W
        img = transform_invert(img_tensor.cpu(), train_transform)
        plt.imshow(img)
        plt.title("LeNet got {} Yuan".format(rmb))
        plt.show()
        plt.pause(0.5)
        plt.close()
