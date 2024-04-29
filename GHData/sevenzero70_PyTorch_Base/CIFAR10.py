# -*- coding: utf-8 -*-
# Author: zero
# Time: 2022.07.16 11:21
import torch
import torchvision
from torch import nn
from torch.nn import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = nn.Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2)
        )
        self.linear_layer = nn.Sequential(
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x


model = Model()
# loss
loss_cross = nn.CrossEntropyLoss()
# optimization
optim = torch.optim.Adam(model.parameters(), lr=1e-3)


# input = torch.ones((64, 3, 32, 32))
# output = model(input)
# writer = SummaryWriter("logs")
# writer.add_graph(model, input)  # 可以观察网络细节图
# writer.close()

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, 1)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        img, target = data
        output_img = model(img)
        result_loss = loss_cross(output_img, target)

        optim.zero_grad()           # 优化器置0，避免每次影响
        result_loss.backward()      # 回传，得到梯度
        optim.step()                # 优化器调优

        running_loss += result_loss
    print("{} epoch loss is: ".format(epoch), running_loss)

# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))  # batch_size = 1, 种类 = 3
# loss_cross = nn.CrossEntropyLoss()
# result = loss_cross(x, y)
