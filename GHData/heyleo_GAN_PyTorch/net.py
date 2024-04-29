import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2)
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(x.size(0), -1)
        x = self.layer_3(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size=100, num_feature=56*56):
        super(Generator, self).__init__()
        self.layer_1 = nn.Linear(in_features=input_size, out_features=num_feature)
        self.layer_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=1), 
            nn.ReLU(True)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=50),
            nn.ReLU(True)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=25),
            nn.ReLU(True)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x

