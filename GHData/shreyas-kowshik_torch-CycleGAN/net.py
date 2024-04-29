import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import cv2
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image

class Discriminator(nn.Module):
        def __init__(self):
                super(Discriminator, self).__init__()
                self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv1 = self.conv_block(3,16)
                self.conv2 = self.conv_block(16,32)
                self.conv3 = self.conv_block(32,64)
                self.conv4 = self.conv_block(64,128)
                self.conv5 = self.conv_block(128,64)
                self.conv6 = self.conv_block(64,32)
                self.conv7 = self.conv_block(32,16)
                self.conv8 = self.final_conv(16,1)

        def conv_block(self,inp,op,padding=1):
                '''
                inp - input channels
                op - output channels
                '''
                return nn.Sequential(nn.Conv2d(inp,op, kernel_size=3, stride=1, padding=padding),nn.BatchNorm2d(op),nn.ReLU(),self.max_pool)

        def final_conv(self,inp,op,padding=1):
                '''
                inp - input channels
                op - output channels
                '''
                return nn.Sequential(nn.Conv2d(inp,op, kernel_size=3, stride=1, padding=padding),nn.BatchNorm2d(op),nn.ReLU(),self.max_pool,nn.Sigmoid())

        def forward(self,x):
                out = self.conv1(x)
                out = self.conv2(out)
                out = self.conv3(out)
                out = self.conv4(out)
                out = self.conv5(out)
                out = self.conv6(out)
                out = self.conv7(out)
                out = self.conv8(out)
                return out