#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.itt as itt

EPOCH = 2
BATCH_SIZE = 512
LR = 0.01
DOWNLOAD = True
DATA = 'dataset/cifar10/'

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut.add_module('sc_conv', nn.Conv2d(in_channel, out_channel, 1))
            self.shortcut.add_module('sc_norm', nn.BatchNorm2d(out_channel))

    def forward(self, x):
        o = self.conv(x)
        o += self.shortcut(x)
        o = F.relu(o)
        return o

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv1 = ResidualBlock(32, 64)
        self.conv2 = ResidualBlock(64, 128)
        self.conv3 = ResidualBlock(128, 256)
        self.conv4 = ResidualBlock(256, 512)
        self.fc1 = nn.Linear(512*8*8, 120)
        self.fc2 = nn.Linear(120, 10)

        self.dropout = nn.Dropout2d()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x

def loadDataset():
    itt.range_push('loadDataset')
    transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = torchvision.datasets.CIFAR10(
        root=DATA,
        train=False,
        transform=transform,
        download=DOWNLOAD,
    )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    itt.range_pop()
    return train_loader, test_loader, classes

def train(train_loader, net, optimizer, epoch):
    itt.range_push('train')
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data
        target = target
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    itt.range_pop()

def test(test_loader, net, optimizer):
    itt.range_push('test')
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data
            target = target
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    itt.range_pop()

def main():
    torch.manual_seed(10)

    train_loader, test_loader, classes = loadDataset()

    net = Net()
    net.share_memory()
    net = net
    optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9)

    for epoch in range(EPOCH):
        itt.range_push('epoch_{}'.format(epoch))
        train(train_loader, net, optimizer, epoch)
        test(test_loader, net, optimizer)
        itt.range_pop()

if __name__ == '__main__':
    with torch.autograd.profiler.emit_itt(enabled=True):
        main()
