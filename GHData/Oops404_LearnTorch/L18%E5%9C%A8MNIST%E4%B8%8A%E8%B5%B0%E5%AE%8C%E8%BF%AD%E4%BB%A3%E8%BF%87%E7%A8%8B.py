# -*- coding: UTF-8-*-
import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/23 10:21
"""
lr = 0.15
gamma = 0.7
epochs = 5
bs = 128

mnist = torchvision.datasets.FashionMNIST(
    root="./data/mnist",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)

batch_data = DataLoader(
    mnist,
    batch_size=bs,
    shuffle=True
)

_input = mnist.data[0].numel()
_output = len(mnist.targets.unique())


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.normalize = nn.BatchNorm2d(num_features=1)
        self.linear1 = nn.Linear(in_features, 128, bias=False)
        self.linear2 = nn.Linear(128, 256, bias=False)
        self.output = nn.Linear(256, out_features, bias=False)

    def forward(self, x):
        x = self.normalize(x)
        # -1代表任意都可，可以理解为占位符。
        # 对数据结构改变，pytorch自动计算-1这个维度应该是多少
        x = x.view(-1, 28 * 28)
        sigma1 = torch.relu(self.linear1(x))
        sigma2 = torch.log_softmax(self.output(torch.relu(self.linear2(sigma1))), dim=1)
        return sigma2


def fit(_net, _batch_data, _lr=0.01, _epochs=5, _gamma=0.1):
    criterion = nn.NLLLoss()
    opt = optim.SGD(_net.parameters(), lr=_lr, momentum=_gamma)
    samples = 0
    correct = 0
    N = 100
    sample_amount = _epochs * len(_batch_data.dataset)
    for epoch in range(_epochs):
        for idx, (x, y) in enumerate(_batch_data):
            # 只能接受Y是一维数据,降维
            y = y.view(x.shape[0])
            sigma = _net.forward(x)
            loss = criterion(sigma, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            # 1,表示在行方向上处理, [1]拿到预测标签
            yhat = torch.max(sigma, 1)[1]
            correct += torch.sum(yhat == y)
            # 累计训练了多少个数据
            samples += x.shape[0]

            if (idx + 1) % N == 0 or (idx + 1) == len(_batch_data):
                print("epoch{}:[{}/{}({:.2f})%], loss:{:.6f}, accuracy:{:.3f}%".format(
                    epoch + 1,
                    samples,
                    sample_amount,
                    100 * (samples / sample_amount),
                    loss.data.item(),
                    100 * (correct / samples)
                ))


torch.manual_seed(996)
net = Model(in_features=_input, out_features=_output)
fit(net, batch_data, _lr=lr, _epochs=epochs, _gamma=gamma)
