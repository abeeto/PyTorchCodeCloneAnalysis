import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import cifar
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

print(f'is CUDA available? {torch.cuda.is_available()}')
# model = torchvision.models.resnet50(pretrained=True)
# print(model)

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class MyResNetPair(nn.Module):
    def __init__(self, channels, input_channels=None, apply_residual=True):
        super(MyResNetPair, self).__init__()
        if input_channels is None:
            input_channels = channels
        self.apply_residual = apply_residual
        self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.apply_residual:
            return x + y
        else:
            return y

    def __repr__(self):
        return f'ConvPair(\n\t{self.conv1}\n\t{self.conv2})'


class MyResNetBlock(nn.Module):
    def __init__(self, channels, num_pairs, input_channel=None):
        super(MyResNetBlock, self).__init__()
        self.num_pairs = num_pairs
        self.input_channel = input_channel
        self.pairs = []
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        for i in range(self.num_pairs):
            if i == 0:
                pair = MyResNetPair(channels, input_channels=input_channel, apply_residual=False)
            else:
                pair = MyResNetPair(channels)
            self.pairs.append(pair)

    def forward(self, x):
        for i, pair in enumerate(self.pairs):
            x = pair(x)
        x = self.max_pool(x)
        return x

    def __repr__(self):
        st = 'ConvBlock(\n'
        for p in self.pairs:
            st += repr(p) + '\n'
        st += ')\n'
        return st


class MyResNet(nn.Module):
    kernel_size = (3, 3)

    def __init__(self, num_pairs: tuple):
        """

        :param channels: number of channels for each block
        :param filters: number of filters for each block
        """
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.block1 = MyResNetBlock(64, num_pairs[0], input_channel=64)
        self.block2 = MyResNetBlock(128, num_pairs[1], input_channel=64)
        self.block3 = MyResNetBlock(256, num_pairs[2], input_channel=128)
        self.block4 = MyResNetBlock(512, num_pairs[3], input_channel=256)
        self.fc = nn.Linear(512 * 3 * 3, 17)  # 17 is class labels

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = self.fc(x)
        return x

# image = Image.open('./13213385.jpg')
# resnet = MyResNet((2, 2, 2, 2))
# img = tfms(image).unsqueeze(0)
# result = resnet(img)
# print(result.shape)
