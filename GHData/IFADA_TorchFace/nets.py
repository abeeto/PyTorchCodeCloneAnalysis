import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Pnet(nn.Module):

    def __init__(self):
        super(Pnet,self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(10,16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.ReLU()
        )
        self.con4_01 = nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.con4_02 = nn.Conv2d(32,4,kernel_size=1,stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        label =F.sigmoid(self.con4_01(x))
        offset = self.con4_02(x)
        return label,offset

class Rnet(nn.Module):
    def __init__(self):
        super(Rnet,self).__init__()
        self.pre_lary = nn.Sequential(
            nn.Conv2d(3,28,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()
        )
        self.conv5 = nn.Linear(64*2*2,256)
        self.prelu5 = nn.PReLU()

        self.conv6_1 = nn.Linear(256,1)
        self.conv6_2 = nn.Linear(256,4)

    def forward(self, x):
      x = self.pre_lary(x)
      x = x.view(x.size(0),-1)#(-1,64*2*2)
      x = self.conv5(x)
      x = self.prelu5(x)
      label = F.sigmoid(self.conv6_1(x))
      offset = self.conv6_2(x)
      return label,offset



class Onet(nn.Module):

    def __init__(self):
       super(Onet,self).__init__()
       self.pre_lary = nn.Sequential(
           nn.Conv2d(3,32,kernel_size=3,stride=1),
           nn.PReLU(),
           nn.MaxPool2d(kernel_size=3,stride=2),

           nn.Conv2d(32,64,kernel_size=3,stride=1),
           nn.PReLU(),
           nn.MaxPool2d(kernel_size=3,stride=2),

           nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
           nn.PReLU(),  # prelu3
           nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
           nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
           nn.PReLU()  # prelu4
       )
       self.conv5 = nn.Linear(128*2*2,256)
       self.prelu5 = nn.PReLU()
       self.conv6_1 = nn.Linear(256,1)
       self.conv6_2 = nn.Linear(256,4)

    def forward(self, x):
        x = self.pre_lary(x)
        x = x.view(x.size(0),-1)
        x = self.conv5(x)
        x = self.prelu5(x)
        label = F.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return label, offset
