import torch
import torch.nn as nn
from torch import flatten
from torch.nn import functional as F


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(3072, 768),
            nn.CELU(),
            nn.Linear(768, 192),
            nn.CELU(),
            nn.Linear(192, 48),
            nn.CELU(),
            nn.Linear(48, 10)
        )
        self.classifier = nn.Sequential()

    def forward(self, x):
        x = flatten(x, 1)
        x = self.features(x)
        out = self.classifier(x)
        return out


class RBF(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))

        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        size = (x.shape[0], self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        out = torch.exp(-1 * distances)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 18, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(18, 48, 6),
            nn.Tanh(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 84),
            nn.Tanh(),
            RBF(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out


# CNN adjusted due to image resolution difference
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LocalResponseNorm(3),
            nn.Conv2d(12, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LocalResponseNorm(3),
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 32, 1, 1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out


# CNN adjusted due to image resolution difference
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out


# CNN adjusted due to image resolution difference
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.celu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.celu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(Block, 16, 5, stride=1)
        self.layer2 = self._make_layer(Block, 32, 5, stride=2)
        self.layer3 = self._make_layer(Block, 64, 5, stride=2)
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.celu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
