from BasicConv import *
from Inception import *
from InceptionAux import *

import torch
import torch.nn as nn 
import torch.nn.functional as F 

class GoogleNet(nn.Module):
    
    def __init__(self, num_classes, dropout=0.2, dropout_aux=0.7):
        super().__init__()
        self.conv1 = BasicConv(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = BasicConv(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.inception4a = Inception(480, 192, 96, 208, 16, 32, 32)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        self.aux1 = InceptionAux(512, num_classes, dropout=dropout_aux)
        self.aux2 = InceptionAux(528, num_classes, dropout=dropout_aux)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
        
        self.init_weights()
        
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
                
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)
        
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)
        aux1 = self.aux1(out)
        
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2 = self.aux2(out)
        
        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        
        out = self.avgpool(out)
        out = out.view(-1, 1024*1*1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out, aux1, aux2