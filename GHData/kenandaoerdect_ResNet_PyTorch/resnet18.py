import torch.nn as nn
import torch


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride):
        super(ResBlk, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
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


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 128, 2),
            ResBlk(128, 128, 1),
            ResBlk(128, 256, 2),
            ResBlk(256, 256, 1),
            ResBlk(256, 512, 2),
            ResBlk(512, 512, 1)
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    res34 = ResNet18()
    sample = torch.randn(8, 3, 227, 227)
    out = res34(sample)
    print(out.shape)