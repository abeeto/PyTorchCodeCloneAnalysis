"""
    Implementation of AlexNet in paper:
    "ImageNet Classification with Deep Convolutional Neural Network"
"""


import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils import data 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
    
    def __init__(self, num_classes=1000):
        super().__init__()
        #   image size = (3*227*227)
        #   input size = (batch_size*3*227*227)
        
        #   convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), #  size: (batch_size*96*55*55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), #    size: (batch_size*96*27*27)
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  #   size: (batch_size*256*27*27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  #   size: (batch_size*256*13*13)
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), #   size: (batch_size*384*13*13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), #   size: (batch_size*384*13*13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), #   size: (batch_size*256*13*13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # size: (batch_size*256*6*6)
        )
        
        #   fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),    #   size: (batch_size*4096)
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),         #   size: (batch_size*4096)
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)   #   size: (batch_size*1000)
        )
        self.init_bias()
        
    def init_bias(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01),
                nn.init.constant_(layer.bias, 0)
        
        #   bias term initialized as 1 for conv 2nd, conv 4th, and conv 5th
        nn.init.constant_(self.conv[4].bias, 1)
        nn.init.constant_(self.conv[10].bias, 1)
        nn.init.constant_(self.conv[12].bias, 1)
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 256*6*6)     #   reduce dimension for fully connected layers
        out = self.fc(out)
        return out
