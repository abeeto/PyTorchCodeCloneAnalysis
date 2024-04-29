from config import args
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # using strided convolutions instead of maxpooling​
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(args.n_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, momentum=args.momentum)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32, momentum=args.momentum)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=args.momentum)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128, momentum=args.momentum)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256, momentum=args.momentum)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=int(args.img_size/32), stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512, momentum=args.momentum)
        self.fc = nn.Linear(512, args.n_classes, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 6 x Conv + Batchnorm + Activation​
        x1 = self.lrelu(self.bn1(self.conv1(x)))
        x2 = self.lrelu(self.bn2(self.conv2(x1)))
        x3 = self.lrelu(self.bn3(self.conv3(x2)))
        x4 = self.lrelu(self.bn4(self.conv4(x3)))
        x5 = self.lrelu(self.bn5(self.conv5(x4)))
        x6 = self.lrelu(self.bn6(self.conv6(x5)))
        x_flat = x6.view(x6.shape[0], x6.shape[1])
        x_out = self.sigmoid(self.fc(x_flat))
        return x_out
