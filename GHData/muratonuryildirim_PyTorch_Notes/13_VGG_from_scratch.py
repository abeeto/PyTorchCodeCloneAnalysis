import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
VGG-16 architecture
Conv2d 3*3, s=1, p=0 (64 feature-maps)
Conv2d 3*3, s=1, p=0 (64 feature-maps) -> max pool
Conv2d 3*3, s=1, p=0 (128 feature-maps) 
Conv2d 3*3, s=1, p=0 (128 feature-maps) -> max pool
Conv2d 3*3, s=1, p=0 (256 feature-maps) 
Conv2d 3*3, s=1, p=0 (256 feature-maps) 
Conv2d 3*3, s=1, p=0 (256 feature-maps) -> max pool
Conv2d 3*3, s=1, p=0 (512 feature-maps)
Conv2d 3*3, s=1, p=0 (512 feature-maps) 
Conv2d 3*3, s=1, p=0 (512 feature-maps) -> max pool
Conv2d 3*3, s=1, p=0 (512 feature-maps)
Conv2d 3*3, s=1, p=0 (512 feature-maps)
Conv2d 3*3, s=1, p=0 (512 feature-maps) -> max pool
Classifier MLP: Linear 4096 (0.5 Dropout) -> Linear 4096 (0.5 Dropout)  -> Linear 1000
'''

VGGs = {"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        }


class VGGNet(nn.Module):
    def __init__(self, architecture, in_channels=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGGs[architecture])
        self.fc_layers = nn.Sequential(nn.Linear(512*7*7, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, 4096),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for i in architecture:
            if type(i) == int:
                out_channels = i
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(i),
                           nn.ReLU()]
                in_channels = i
            else:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


x = torch.randn(1, 3, 224, 224)
model = VGGNet('VGG16')
print(model(x).shape)
