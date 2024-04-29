# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import transforms, models

import os
from models.net import Resnet50

from config import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, criterion, optimizer, train_loader, epoch, cfg):
    net.train()
    train_loss = 0.0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 计算
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 打印
        if (batch_idx + 1) % cfg.interval == 0:
            print("[Epoch:{}/{}]:[Batch:{}/{}]:Loss:{:.3f}|Acc:{:.3f}%".format(
                epoch, cfg.num_epochs, batch_idx, len(train_loader), train_loss / cfg.interval, 100.*correct/total))
            train_loss = 0.0
            total = 0
            correct = 0


def test(net, criterion, test_loader):
    net.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # 计算
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印
        print("Test Loss:{:.3f}|Acc:{:.3f}%".format(test_loss / len(test_loader), 100.*correct/total))

def main():
    # 保存
    if not os.path.exists(cfg.output):
        os.mkdir(cfg.output)

    # 训练集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(root=cfg.root, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # 测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = torchvision.datasets.CIFAR10(root=cfg.root, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 网络
    net = Resnet50(num_classes=len(cfg.classes))
    net = net.to(device)

    # fp16
    if cfg.fp16:
        from src.fp16util import network_to_half
        net = network_to_half(net)

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, cfg.num_epochs+1):
        train(net, criterion, optimizer, train_loader, epoch, cfg)
        if epoch % cfg.valinterval == 0:
            test(net, criterion, test_loader)
            if not cfg.fp16:
                torch.save(net.state_dict(), f"./{cfg.output}/resnet50-{epoch}-ckpt.t7")
            else:
                torch.save(net.state_dict(), f"./{cfg.output}/resnet50-{epoch}-fp16-ckpt.t7")




if __name__ == '__main__':
    main()

