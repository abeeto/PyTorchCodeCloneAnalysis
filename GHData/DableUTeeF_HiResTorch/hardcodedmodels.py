import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class AddBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, blocks, size):
        super(AddBlock, self).__init__()
        self.size = size
        self.blocks = []
        for _ in range(blocks):
            self.blocks.append(Block(in_planes, planes).cuda())

    def forward(self, x):
        out = self.blocks[0](x)
        path = F.avg_pool2d(out, self.size)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](x)
            path += F.avg_pool2d(out, self.size)
        return path


class Block(nn.Module):
    def __init__(self, in_planes, planes):
        super(Block, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(in_planes, plane1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, plane):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(plane)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(plane)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = F.relu(out)
        return out


class HiResA(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResA, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1_1 = Block(32, [128, 32, 32])
        self.layer1_2 = Block(32, [128, 32, 32])
        self.layer1_3 = Block(32, [128, 32, 32])
        self.layer2_1 = Block(32, [256, 64, 64])
        self.layer2_2 = Block(64, [256, 64, 64])
        self.layer2_3 = Block(64, [256, 64, 64])
        self.layer3_1 = Block(64, [512, 128, 128])
        self.layer3_2 = Block(128, [512, 128, 128])
        self.layer3_3 = Block(128, [512, 128, 128])
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        out = self.layer1_1(out)
        path += F.avg_pool2d(out, 28)
        out = self.layer1_2(out)
        path += F.avg_pool2d(out, 28)
        out = self.layer1_3(out)
        # path += F.avg_pool2d(out, 28)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.max_pool2d(out, 2)

        out = self.layer2_1(out)
        path += F.avg_pool2d(out, 14)
        out = self.layer2_2(out)
        path += F.avg_pool2d(out, 14)
        out = self.layer2_3(out)
        # path += F.avg_pool2d(out, 14)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        # path += F.avg_pool2d(out, 14)

        out = F.max_pool2d(out, 2)
        out = self.layer3_1(out)
        path += F.avg_pool2d(out, 7)
        out = self.layer3_2(out)
        path += F.avg_pool2d(out, 7)
        out = self.layer3_3(out)
        # path += F.avg_pool2d(out, 7)
        path = torch.cat((path, F.avg_pool2d(out, 7)), 1)
        # path += F.avg_pool2d(out, 7)

        # out = F.avg_pool2d(path, 4)
        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out


class HiResB(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResB, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_1 = BasicBlock(64, [64, 64])
        self.layer1_2 = BasicBlock(64, [64, 64])
        self.layer1_3 = BasicBlock(64, [64, 64])
        self.layer2_1 = BasicBlock(64, [128, 128])
        self.layer2_2 = BasicBlock(128, [128, 128])
        self.layer2_3 = BasicBlock(128, [128, 128])
        self.layer3_1 = BasicBlock(128, [256, 256])
        self.layer3_2 = BasicBlock(256, [256, 256])
        self.layer3_3 = BasicBlock(256, [256, 256])
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        out = self.layer1_1(out)
        path += F.avg_pool2d(out, 28)
        out = self.layer1_2(out)
        path += F.avg_pool2d(out, 28)
        out = self.layer1_3(out)
        # path += F.avg_pool2d(out, 28)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.max_pool2d(out, 2)

        out = self.layer2_1(out)
        path += F.avg_pool2d(out, 14)
        out = self.layer2_2(out)
        path += F.avg_pool2d(out, 14)
        out = self.layer2_3(out)
        # path += F.avg_pool2d(out, 14)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        # path += F.avg_pool2d(out, 14)

        out = F.max_pool2d(out, 2)
        out = self.layer3_1(out)
        path += F.avg_pool2d(out, 7)
        out = self.layer3_2(out)
        path += F.avg_pool2d(out, 7)
        out = self.layer3_3(out)
        # path = torch.cat((path, F.avg_pool2d(out, 7)), 1)
        path += F.avg_pool2d(out, 7)

        # out = F.avg_pool2d(path, 4)
        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out


class HiResC(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResC, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_1 = BasicBlock(64, 64)
        self.layer1_2 = BasicBlock(64, 64)
        self.layer1_3 = BasicBlock(64, 64)
        self.layer2_1 = BasicBlock(64, 128)
        self.layer2_2 = BasicBlock(128, 128)
        self.layer2_3 = BasicBlock(128, 128)
        self.layer3_1 = BasicBlock(128, 256)
        self.layer3_2 = BasicBlock(256, 256)
        self.layer3_3 = BasicBlock(256, 256)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        out = self.layer1_1(out)
        path += F.avg_pool2d(out, 28)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer1_2(out)
        path += F.avg_pool2d(out, 28)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer1_3(out)
        # path += F.avg_pool2d(out, 28)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.relu(out)
        path = F.relu(path)
        out = F.max_pool2d(out, 2)

        out = self.layer2_1(out)
        path += F.avg_pool2d(out, 14)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer2_2(out)
        path += F.avg_pool2d(out, 14)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer2_3(out)
        # path += F.avg_pool2d(out, 14)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        out = F.relu(out)
        path = F.relu(path)
        # path += F.avg_pool2d(out, 14)

        out = F.max_pool2d(out, 2)
        out = self.layer3_1(out)
        path += F.avg_pool2d(out, 7)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer3_2(out)
        path += F.avg_pool2d(out, 7)
        out = F.relu(out)
        path = F.relu(path)
        out = self.layer3_3(out)
        # path = torch.cat((path, F.avg_pool2d(out, 7)), 1)
        path += F.avg_pool2d(out, 7)
        path = F.relu(path)

        # out = F.avg_pool2d(path, 4)
        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out


class HiResD(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(HiResD, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_1 = BasicBlock(64, 64)
        self.layer1_2 = BasicBlock(64, 64)
        self.layer1_3 = BasicBlock(64, 64)
        self.layer2_1 = BasicBlock(64, 128)
        self.layer2_2 = BasicBlock(128, 128)
        self.layer2_3 = BasicBlock(128, 128)
        self.layer3_1 = BasicBlock(128, 256)
        self.layer3_2 = BasicBlock(256, 256)
        self.layer3_3 = BasicBlock(256, 256)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        path = F.avg_pool2d(out, 28)
        out = self.layer1_1(out)
        path += F.avg_pool2d(out, 28)
        out = F.relu(out)
        out = self.layer1_2(out)
        path += F.avg_pool2d(out, 28)
        out = F.relu(out)
        out = self.layer1_3(out)
        # path += F.avg_pool2d(out, 28)
        path = torch.cat((path, F.avg_pool2d(out, 28)), 1)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.layer2_1(out)
        path += F.avg_pool2d(out, 14)
        out = F.relu(out)
        out = self.layer2_2(out)
        path += F.avg_pool2d(out, 14)
        out = F.relu(out)
        out = self.layer2_3(out)
        # path += F.avg_pool2d(out, 14)
        path = torch.cat((path, F.avg_pool2d(out, 14)), 1)
        out = F.relu(out)
        # path += F.avg_pool2d(out, 14)

        out = F.max_pool2d(out, 2)
        out = self.layer3_1(out)
        path += F.avg_pool2d(out, 7)
        out = F.relu(out)
        out = self.layer3_2(out)
        path += F.avg_pool2d(out, 7)
        out = F.relu(out)
        out = self.layer3_3(out)
        # path = torch.cat((path, F.avg_pool2d(out, 7)), 1)
        path += F.avg_pool2d(out, 7)
        path = F.relu(path)

        # out = F.avg_pool2d(path, 4)
        out = path.view(path.size(0), -1)
        out = self.linear(out)
        return out
