import torch.nn as nn
import torch


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Mymodel(nn.Module):
    def __init__(self, num_class=10):
        super(Mymodel,self).__init__()

        self.unit1 = Unit(in_channels=3,out_channels=32)    # 32*28*28
        self.unit2 = Unit(in_channels=32,out_channels=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)         # 32*14*14

        self.unit3 = Unit(in_channels=32,out_channels=64)   # 64*14*14
        self.unit4 = Unit(in_channels=64,out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)         # 64*7*7 = 3136


        self.net = nn.Sequential(self.unit1,self.unit2,self.maxpool1,
                                 self.unit3,self.unit4,self.maxpool2)

        self.fc = nn.Linear(in_features=3136,out_features=10)

    def forward(self, input):
        output = self.net(input)
        output = output.view(input.shape[0],-1)
        output = self.fc(output)
        return output



