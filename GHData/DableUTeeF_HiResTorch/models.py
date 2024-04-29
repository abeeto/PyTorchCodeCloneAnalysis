import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class AddBlock(nn.Module):
    def __init__(self, in_planes, plane, stride, pool_size):
        super(AddBlock, self).__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)

    def forward(self, z):
        x, w = z[0], z[1]
        out = self.conv1(x)
        out = self.bn1(out)
        w += F.avg_pool2d(out, self.pool_size)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out, w


class CatBlock(nn.Module):
    def __init__(self, in_planes, plane, stride, pool_size):
        super(CatBlock, self).__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)

    def forward(self, z):
        x, w = z[0], z[1]
        out = self.conv1(x)
        out = self.bn1(out)
        w = torch.cat((w, F.avg_pool2d(out, self.pool_size)), 1)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out, w


class HiResA(nn.Module):
    def __init__(self, block, num_blocks, initial_kernal=64, num_classes=10):
        super(HiResA, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, initial_kernal, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        for _ in range(num_blocks[0]):
            self.layer1.append(block(initial_kernal, [initial_kernal*4, initial_kernal, initial_kernal]))
        for _ in range(num_blocks[1]):
            self.layer2.append(block(initial_kernal * 2, [initial_kernal*4 if _ == 0 else initial_kernal*8
                , initial_kernal*2, initial_kernal*2]))
        for _ in range(num_blocks[2]):
            self.layer3.append(block(initial_kernal * 4, [initial_kernal*8 if _ == 0 else initial_kernal*16
                , initial_kernal*4, initial_kernal*4]))
        self.linear = nn.Linear(initial_kernal*4, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        for i in range(self.num_blocks[0] - 1):
            out = self.layer1[i](out)
            path += F.avg_pool2d(out, 28)

        out = self.layer1[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer2[i](out)
            path += F.avg_pool2d(out, 14)
        out = self.layer2[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        out = F.max_pool2d(out, 2)

        for i in range(self.num_blocks[1] - 1):
            out = self.layer3[i](out)
            path += F.avg_pool2d(out, 7)
        out = self.layer3[-1](out)
        path = torch.cat((path, F.avg_pool2d(out, 7)), 1)

        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out


class HiResC(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResC, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, pool_size=32)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, pool_size=16)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, pool_size=8)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride, pool_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride == 2:
                layers.append(CatBlock(self.in_planes, planes, stride, pool_size))
            else:
                layers.append(AddBlock(self.in_planes, planes, stride, pool_size))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        w = F.avg_pool2d(out, 32)
        out = F.relu(out)
        z = self.layer1((out, w))
        z = self.layer2(z)
        out, w = self.layer3(z)
        w = F.relu(w)
        out = w.view(w.size(0), -1)
        out = self.linear(out)
        return out
