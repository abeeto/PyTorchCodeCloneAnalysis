import torch
import torch.nn as nn
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1, shortcut = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3, stride = 1, padding= 1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        residual = x if self.shortcut == None else self.shortcut(x)
        out += residual

        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, expansion = 2, shortcut = None):
        super(Bottleneck, self).__init__()
        bottle_planes = inplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, 3, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = shortcut

    def forward(self, x):
        residual = x if self.shortcut == None else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out += residual

        return out


class Resnet34(nn.Module):
    def __init__(self, numclasses = 1000):
        super(Resnet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layers_(64, 64, 3)
        self.layer2 = self._make_layers_(64, 128, 4, 2)
        self.layer3 = self._make_layers_(128, 256, 6, 2)
        self.layer4 = self._make_layers_(256, 512, 3, 2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, numclasses)


    def _make_layers_(self, inplanes, planes, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride),
            nn.BatchNorm2d(planes)
        )
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, shortcut))

        for i in range(1, block_num):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet101(nn.Module):
    def __init__(self, numclasses = 1000):
        super(Resnet101, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layers1(64, 256, 3)
        self.layer2 = self._make_layers2(256, 512, 4, 2)
        self.layer3 = self._make_layers2(512, 1024, 23, 2)
        self.layer4 = self._make_layers2(1024, 2048, 3, 2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, numclasses)


    def _make_layers1(self, inplanes, planes, block_num, stride=1, expansion = 2):
        shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride),
            nn.BatchNorm2d(planes)
        )
        layers = []
        layers.append(Bottleneck(inplanes, planes, stride, 1, shortcut))

        for i in range(1, block_num):
            layers.append(Bottleneck(planes, planes, 1, 4))

        return nn.Sequential(*layers)

    def _make_layers2(self, inplanes, planes, block_num, stride=1, expansion = 2):
        shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride),
            nn.BatchNorm2d(planes)
        )
        layers = []
        layers.append(Bottleneck(inplanes, planes, stride = stride, expansion = 2, shortcut = shortcut))

        for i in range(1, block_num):
            layers.append(Bottleneck(planes, planes, 1, 4))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        print(x.size())
        x = self.layer1(x)
        print(x.size())
        x = self.layer2(x)
        print(x.size())
        x = self.layer3(x)
        print(x.size())
        x = self.layer4(x)
        print(x.size())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = Resnet101()
    x = torch.randn((2, 3, 224, 224))
    print(net(x).size())
