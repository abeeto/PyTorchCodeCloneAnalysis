from torch import nn
import torch
from torchsummary import summary


class ResBlock(nn.Module):


    def __init__(self,input,out,downsample):
        super(ResBlock, self).__init__()
        if (downsample==True):
            self.layer1 = nn.Sequential(
                nn.Conv2d(input, out, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(out),
                nn.ReLU()
            )
            self.skipconection = nn.Sequential(
                nn.Conv2d(input, out, kernel_size=1, stride=2),
                nn.BatchNorm2d(out)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(input, out, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out),
                nn.ReLU()
            )
            self.skipconection = nn.Sequential()

        self.layer2 = nn.Sequential(
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )

    def forward(self, input):
        skip = self.skipconection(input)
        out = self.layer1(input)
        out = self.layer2(out)
        out = out+skip
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_chanel, resblock,num_class):


        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanel, out_channels=64, kernel_size=7, stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )
        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )


    def forward(self,x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classify(out)
        return out


resnet18 = ResNet(3, ResBlock, num_class=3)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet18, (3, 224, 224))




