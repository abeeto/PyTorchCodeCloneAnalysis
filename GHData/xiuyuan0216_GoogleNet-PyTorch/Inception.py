from BasicConv import *
import torch 
import torch.nn as nn


class Inception(nn.Module):
    
    def __init__(self, in_channels, conv1x1, conv3x3first, conv3x3, conv5x5first, conv5x5, pool):
        super().__init__()
        self.branch1 = BasicConv(in_channels, conv1x1, kernel_size=1, stride=1)
        
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, conv3x3first, kernel_size=1, stride=1),
            BasicConv(conv3x3first, conv3x3, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, conv5x5first, kernel_size=1, stride=1),
            BasicConv(conv5x5first, conv5x5, kernel_size=5, stride=1, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_channels, pool, kernel_size=1, stride=1)
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        out = [out1, out2, out3, out4]
        out = torch.concat(out, 1)
        return out