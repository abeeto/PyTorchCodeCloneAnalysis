import torch
import torch.nn as nn
import torch.nn.functional as F


class WatoNet(nn.Module):
    def __init__(self, in_channels, n_classes):  # constructor
        super(WatoNet, self).__init__()  # parent constructor

        self.drop = nn.Dropout2d(p=0.4)

        # in_channels, out_channels, kernel_size
        # self.conv0 = nn.Conv2d(3, 32, 5, padding=2)
        # self.conv0_bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(in_channels, 48, 5, padding=2)
        self.conv1_bn = nn.GroupNorm(4, 48)
        self.conv2 = nn.Conv2d(48, 64, 5, padding=2, stride=2)
        self.conv2_bn = nn.GroupNorm(4, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_bn = nn.GroupNorm(8, 128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv4_bn = nn.GroupNorm(16,256)
        # self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        # self.conv5_bn = nn.BatchNorm2d(256)

        # and now the decoder layers!

        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv6_bn = nn.GroupNorm(8, 128)
        # upsample here
        self.conv7 = nn.Conv2d(128+128, 128, 3, padding=1)
        self.conv7_bn = nn.GroupNorm(4, 128)
        self.conv8 = nn.Conv2d(128, 64, 5, padding=2)
        self.conv8_bn = nn.GroupNorm(4, 64)
        # upsample here
        self.conv9 = nn.Conv2d(64+48, 64, 7, padding=3)
        self.conv9_bn = nn.GroupNorm(8, 64)
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv10_bn = nn.GroupNorm(4, 32)

        self.head = nn.Conv2d(32, n_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = F.leaky_relu(self.conv0_bn(self.conv0(x)))
        # x = self.drop(x)
        x0 = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x0)))
        x = self.drop(x)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x1 = self.drop(x)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x1)))
        x = self.drop(x)
        # x = F.leaky_relu(self.conv5_bn(self.conv5(x)))
        # x = self.drop(x)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = self.drop(x)
        x = F.upsample(x, scale_factor=2)
        x = torch.cat((x, x1), dim=1)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))
        x = self.drop(x)
        x = F.leaky_relu(self.conv8_bn(self.conv8(x)))
        x = self.drop(x)
        x = F.upsample(x, scale_factor=2)
        x = torch.cat((x, x0), dim=1)
        x = F.leaky_relu(self.conv9_bn(self.conv9(x)))
        x = self.drop(x)
        x = F.leaky_relu(self.conv10_bn(self.conv10(x)))
        x = self.head(x)
        x = F.softmax(x, dim=1)  # apply softmax along dim 1 (dim 0 is the different batches)
        return x


