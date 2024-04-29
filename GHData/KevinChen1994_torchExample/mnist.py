# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2022/2/8 23:22

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
test_set = datasets.MNIST('data', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class LogisticsRegression(nn.Module):
    def __init__(self):
        super(LogisticsRegression, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)  # 第一个维度的值就是batch_size
        '''
        使用0填充
        output_height=[input_height/stride_height]  
        output_width=[input_width/stride_width]
        不使用0填充 
        output_height=[(input_height-filter_height+1)/stride_height]  
        output_width=[(input_width-filter_width+1)/stride_width]
        '''
        out = self.conv1(x)  # [batch_size, 1, 28, 28] -> [batch_size, 10, 24, 24] 默认是不使用0填充
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # [batch_size, 10, 24, 24] -> [batch_size, 10, 12, 12]
        out = self.conv2(out)  # [batch_size, 10, 12, 12] -> [batch_size, 20, 10, 10]
        out = F.relu(out)
        out = out.view(in_size, -1)  # [batch_size, 20, 10, 10] -> [batch_size, 20*10*10]
        out = self.fc1(out)  # [batch_size, 200] -> [batch_size, 500]
        out = F.relu(out)
        out = self.fc2(out)  # [batch_size, 500] -> [batch_size, 10]
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out


# model = LogisticsRegression().to(device=DEVICE)
model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
