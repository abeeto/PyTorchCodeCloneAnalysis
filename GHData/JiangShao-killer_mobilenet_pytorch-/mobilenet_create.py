# -*- coding: UTF-8 -*-  
# @Time : 2022/2/13 12:56
# @Project:python_pytorch
# @Author : 江小强
# @File : mobilenet_create.py
# @Software : PyCharm

import torch
from torch import nn


def depthwise_conv(inputc, outputc, stride1=1):
    return nn.Sequential(
            nn.Conv2d(in_channels = inputc, out_channels = inputc,
                      kernel_size = 3, stride = stride1,padding =1,groups = inputc),
            nn.BatchNorm2d(inputc),
            nn.ReLU6(inplace = True),
            nn.Conv2d(in_channels = inputc, out_channels = outputc,
                      kernel_size = 1, stride = 1,padding = 0),
            nn.BatchNorm2d(outputc),
            nn.ReLU6(inplace = True)
    )


def conv_bn(inputc, outputc, stride):
    return nn.Sequential(
            nn.Conv2d(in_channels = inputc, out_channels = outputc,
                      kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(outputc),
            nn.ReLU6(inplace = True)
    )


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.stage1 = nn.Sequential(
                conv_bn(3, 32, 2),
                depthwise_conv(32, 64, 1),
                depthwise_conv(64, 128, 2),
                depthwise_conv(128, 128, 1),
                depthwise_conv(128, 256, 2),
                depthwise_conv(256, 256, 1),
                depthwise_conv(256, 512, 2)
        )
        self.conv = conv_bn(3, 32, 2)
        self.conv_dw1 = depthwise_conv(32, 64, 1)
        self.conv_dw2 = depthwise_conv(64, 128, 2)
        self.conv_dw3 = depthwise_conv(128, 128, 1)
        self.conv_dw4 = depthwise_conv(128, 256, 2)
        self.conv_dw5 = depthwise_conv(256, 256, 1)
        self.conv_dw6 = depthwise_conv(256, 512, 2)

        self.five_condw = nn.Sequential(
                depthwise_conv(512, 512, 1),
                depthwise_conv(512, 512, 1),
                depthwise_conv(512, 512, 1),
                depthwise_conv(512, 512, 1),
                depthwise_conv(512, 512, 1)
        )

        self.stage3 = nn.Sequential(
                depthwise_conv(512, 1024, 2),
                depthwise_conv(1024, 1024, 1)
        )

        self.conv_dw8 = depthwise_conv(512, 1024, 2)
        self.conv_dw9 = depthwise_conv(1024, 1024, 1)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        in_size = x.size(0)
        model = nn.Sequential(
                self.conv,
                self.conv_dw1,
                self.conv_dw2,
                self.conv_dw3,
                self.conv_dw4,
                self.conv_dw5,
                self.conv_dw6,
                self.five_condw,
                self.conv_dw8,
                self.conv_dw9,
                self.avgpool,
        )
        x = model(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = MobileNet()
    model = model.to(device)
    summary(model, input_size = (3, 224, 224))
    torch.save(model,'mobilenet.pth')

