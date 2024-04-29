import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class Conv2d(nn.Module):
    def __init__(self,in_c,out_c,k_size,stride=1,activate=True,padding=1,In=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.In = nn.InstanceNorm2d(out_c, affine=True) if In else None
        self.relu = nn.ReLU(inplace=True) if activate else None

    def forward(self,x):
        x = self.conv(x)
        x = self.In(x) if self.In else x
        x = self.relu(x) if self.relu else x
        return x


class DeConv2d(nn.Module):
    def __init__(self, in_c, out_c,k_size,stride,padding=0, activate=True, In=False):
        super(DeConv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=k_size, stride=stride)
        self.In = nn.InstanceNorm2d(out_c, affine=True) if In else None
        self.relu = nn.ReLU(inplace=True) if activate else None

    def forward(self, x):
        x = self.deconv(x)
        x = self.In(x) if self.In else x
        x = self.relu(x) if self.relu else x
        return x



class SAModule(nn.Module):

    def __init__(self,in_c,out_c,In=True,middleLayer=True):
        super(SAModule, self).__init__()
        self.branch_1 = Conv2d(in_c,out_c//4,1,padding=0,In=False)
        if not middleLayer:
            self.branch_2 = Conv2d(in_c,out_c//4,3,padding=1,In=True)
            self.branch_3 = Conv2d(in_c,out_c//4,5,padding=2,In=True)
            self.branch_4 = Conv2d(in_c,out_c//4,7,padding=3,In=True)
        else:
            self.branch_2 = nn.Sequential(
                Conv2d(in_c,out_c//2,1,padding=0,In=True),
                Conv2d(out_c//2,out_c//4,3,padding=1,In=True)
            )
            self.branch_3 = nn.Sequential(
                Conv2d(in_c,out_c//2,1,padding=0,In=True),
                Conv2d(out_c//2,out_c//4,5,padding=2,In=True)
            )
            self.branch_4 = nn.Sequential(
                Conv2d(in_c,out_c//2,1,padding=0,In=True),
                Conv2d(out_c//2,out_c//4,7,padding=3,In=True)
            )

    def forward(self,x):
        out1 = self.branch_1(x)
        out2 = self.branch_2(x)
        out3 = self.branch_3(x)
        out4 = self.branch_4(x)
        out = torch.cat([out1, out2, out3, out4], 1)
        return out



class SANet(nn.Module):

    '''
    -Implementation of <<Scale Aggregation Network for Accurate and Efficient Crowd Counting>>
    '''

    def __init__(self):
        super(SANet, self).__init__()
        self.fme_module = nn.Sequential(
            SAModule(3, 16, middleLayer=False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            SAModule(16, 32, middleLayer=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SAModule(32, 32, middleLayer=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SAModule(32, 16, middleLayer=True)
        )

        self.dme_module = nn.Sequential(
            Conv2d(16, 64, 9, padding=4, In=True),
            DeConv2d(64, 64, k_size=2, stride=2, In=True),
            Conv2d(64, 32, 7, padding=3, In=True),
            DeConv2d(32, 32, k_size=2, stride=2, In=True),
            Conv2d(32, 16, 5, padding=2, In=True),
            DeConv2d(16, 16, k_size=2, stride=2, In=True),
            Conv2d(16, 16, 3, padding=1, In=True),
            Conv2d(16, 16, 5, padding=2, In=True)
        )

        self.out_layer = Conv2d(16,1,1,padding=0, In=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.fme_module(x)
        x = self.dme_module(x)
        x = self.out_layer(x)
        return x

