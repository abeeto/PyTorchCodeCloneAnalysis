import torch
import torch.nn as nn


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(True),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out)
        )

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, input):
        out = self.block(input)+self.extra(input)
        return out


class ResNet(nn.Module):

    def __init__(self, conv2_num, conv3_num, conv4_num, conv5_num):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = nn.Sequential()
        self.conv2_x.add_module('conv2_0', ResBlk(64, 256))
        for i in range(conv2_num-1):
            self.conv2_x.add_module('conv2_%d'%(i+1), ResBlk(256, 256))

        self.conv3_x = nn.Sequential()
        self.conv3_x.add_module('conv3_0', ResBlk(256, 512, 2))
        for i in range(conv3_num-1):
            self.conv3_x.add_module('conv3_%d'%(i+1), ResBlk(512, 512))

        self.conv4_x = nn.Sequential()
        self.conv4_x.add_module('conv4_0', ResBlk(512, 1024, 2))
        for i in range(conv4_num-1):
            self.conv4_x.add_module('conv4_%d'%(i+1), ResBlk(1024, 1024))

        self.conv5_x = nn.Sequential()
        self.conv5_x.add_module('conv5_0', ResBlk(1024, 2048, 2))
        for i in range(conv5_num-1):
            self.conv5_x.add_module('conv5_%d'%(i+1), ResBlk(2048, 2048))

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 10)


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet50():
    return ResNet(3, 4, 6, 3)


def resnet101():
    return ResNet(3, 4, 23, 3)


def resnet152():
    return ResNet(3, 8, 36, 3)

if __name__ == '__main__':
    net = resnet152()
    out = net(torch.randn(2, 3, 227, 227))
    print(out.shape)