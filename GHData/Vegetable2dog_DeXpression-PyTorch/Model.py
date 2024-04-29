import torch
import torch.nn as nn
import torch.nn.functional as F


# image pre-processor block
class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(size=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.lrn(x)
        # print("BLOCK 1 = ", x.size())
        return x


# feature extractor block 1
class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1)
        self.pool2a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, padding=1)
        self.conv2c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.pool2b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.padding = nn.ZeroPad2d((72, 72, 0, 0))

    def forward(self, x1):
        x2 = x1
        x1 = F.relu(self.conv2a(x1))
        x1 = F.relu(self.conv2b(x1))
        x2 = self.pool2a(x2)
        x2 = F.relu(self.conv2c(x2))
        x2 = F.pad(x2, (0, 0, 0, 0, 72, 72))
        x3 = torch.cat((x1, x2))
        x3 = self.pool2b(x3)
        # print("BLOCK 2 = ", x3.size())
        return x3


# feature extractor block 2
class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.conv3a = nn.Conv2d(in_channels=208, out_channels=96, kernel_size=1)
        self.pool3a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(in_channels=208, out_channels=64, kernel_size=1)
        self.pool3b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.125)
        self.fc1 = nn.Linear(208*13*13*4, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x1):
        x2 = x1
        x1 = F.relu(self.conv3a(x1))
        x2 = self.pool3a(x2)
        x1 = F.relu(self.conv3b(x1))
        x2 = F.relu(self.conv3c(x2))
        x2 = F.pad(x2, (0, 0, 0, 0, 72, 72))
        x3 = torch.cat((x1, x2))
        x3 = self.pool3b(x3)
        # print("BLOCK 3 = ", x3.size())
        # x3 = self.dropout(x3)
        x3 = x3.view(-1, 208*13*13*4)
        x3 = F.relu(self.fc1(x3))
        x3 = F.relu(self.fc2(x3))
        x3 = F.log_softmax(self.fc3(x3))   # we want to use softmax over here
        return x3


block1 = Block1()
block2 = Block2()
block3 = Block3()
# printing size of entire network
net_total_params = sum(p.numel() for p in block1.parameters()) + sum(p.numel() for p in block2.parameters()) + \
                   sum(p.numel() for p in block3.parameters())
print(net_total_params)
