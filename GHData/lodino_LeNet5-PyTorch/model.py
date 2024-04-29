import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = Conv(in_ch=1, out_ch=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = Conv(in_ch=6, out_ch=16, kernel_size=5, stride=1)
        self.conv3 = Conv(in_ch=16, out_ch=120, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lenet5 = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            Flatten(),
            FCL(120, 84, activation="tanh"),
            FCL(84, 10, activation="softmax")
        )
        print(self.lenet5)

    def forward(self, x):
        x = self.lenet5(x)
        return x


class Conv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, bn=True, padding=0):
        super(Conv, self).__init__()
        if not bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                # functools.partial(nn.BatchNorm2d, affine=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class FCL(nn.Module):

    def __init__(self, in_dim, out_dim, bn=True, activation="tanh"):
        super(FCL, self).__init__()

        if activation == "tanh":
            self.activation_layer = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_layer = nn.Sigmoid()
        elif activation == "relu":
            self.activation_layer = nn.ReLU(True)
        elif activation == "softmax":
            self.activation_layer = nn.Softmax()

        if not bn:
            self.fcl = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                self.activation_layer
            )
        else:
            self.fcl = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # functools.partial(nn.BatchNorm1d, affine=True),
                nn.BatchNorm1d(out_dim),
                self.activation_layer
            )

    def forward(self, x):
        x = self.fcl(x)
        return x


# From: https://forums.fast.ai/t/flatten-layer-of-pytorch/4639/5
class Flatten(nn.Module):

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x