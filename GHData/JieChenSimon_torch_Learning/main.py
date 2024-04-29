'''
Description: 
Version: 2.0
Autor: CHEN JIE
Date: 2020-09-21 08:34:48
LastEditors: CHEN JIE
LastEditTime: 2020-09-25 09:04:15
language: Python
Deep learning framework: Pytorch1.4.0
'''

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from lenet5 import Lenet5
from torch import nn, optim


def main():
    batchsz= 32
    # cifar为torch自带的数据集，使用.来调用目标数据集，第一个位置为路径，如本地没有，则配合download=true在线下载到改路径里；
    # 第二个参数为“是否为训练集”，为布尔值，true的话则下载训练集，false则下载验证集。即训练集和测试集合是分开下载的。
    # 第三个参数trainsform，是将数据集中的图片根据具体的实验模型进行变化，为函数格式
    cifar_train = datasets.CIFAR10('CIFAR10', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    #加载训练/测试集
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('CIFAR10', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)

    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:' , x.shape, 'label:', label.shape)
    
    device = torch.device('cuda')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    
    model.train()
    for epoch in range(1000):
        for batchsz, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)

            logits = model(x)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(epoch, loss.item())



    model.eval()
    with torch.no_grad():
        #test
        total_correct = 0
        total_num = 0
        for x, label in cifar_test:
            x, label = x.to(device), label.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        
        acc = total_correct / total_num
        print(epoch, acc)

    
        






if __name__ == "__main__":
    main()