import torch
import torch.nn as nn


class DepthwiseConv(nn.Module):

    def __init__(self, ch_in, ch_out, stride):
        super(DepthwiseConv, self).__init__()

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(True),

        )

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv_3x3(x)
        out = self.conv_1x1(out)
        return out


class Mobilenet(nn.Module):

    def __init__(self):
        super(Mobilenet, self).__init__()

        self.conv = nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
             nn.BatchNorm2d(32),
             nn.ReLU(True),
         )

        self.conv_dw = nn.Sequential(
             DepthwiseConv(32, 64, 1),
             DepthwiseConv(64, 128, 2),
             DepthwiseConv(128, 128, 1),
             DepthwiseConv(128, 256, 2),
             DepthwiseConv(256, 256, 1),
             DepthwiseConv(256, 512, 2),
             DepthwiseConv(512, 512, 1),
             DepthwiseConv(512, 512, 1),
             DepthwiseConv(512, 512, 1),
             DepthwiseConv(512, 512, 1),
             DepthwiseConv(512, 512, 1),
             DepthwiseConv(512, 1024, 2),
             DepthwiseConv(1024, 1024, 1)
         )

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    net = Mobilenet()
    a = torch.randn(8, 3, 224, 224)
    sample = net(a)
    print(sample.shape)

