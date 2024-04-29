# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:14:08 2021

@author: mcgoug01
"""

import torch
import torch.nn as nn

#create dictionaries for activations, optimisation schemes

#need to correctly switch between .train() and .eval() in train-time
#to turn off the effects of dropout (among other things like batch normalisation)

init = torch.nn.init.kaiming_uniform_

class deconv(nn.Module):
    def __init__(self,in_c,out_c,drop=0.6):
      super(deconv,self).__init__()
      self.actv = nn.LeakyReLU(0.1)
      self.pad = nn.ConstantPad3d((1,2,1,2,0,0),0)
      
      self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c,
                             kernel_size=3)
      
      self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c,
                             kernel_size=3, stride=2)
      init(self.conv1.weight)
      init(self.conv2.weight)
      self.drop = nn.Dropout(p=drop)
    
    def forward(self,in_):
        in_ = self.drop(self.conv1(self.pad(in_)))
        return self.actv(self.drop(self.conv2(in_)))


class upconv(nn.Module):
    def __init__(self,in_c,out_c,drop=0.6):
      super(upconv,self).__init__()
      self.actv = nn.LeakyReLU(0.1)
      self.pad = nn.ConstantPad3d((1,1,1,1,0,0),0)
      
      self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c,
                             kernel_size=3)
      self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c,
                             kernel_size=3)
      init(self.conv1.weight)
      init(self.conv2.weight)
      self.drop = nn.Dropout(p=drop)
      self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear',align_corners=False)
    
    
    def forward(self,in_conv,in_skip):
        in_conv = self.conv1(self.pad(self.up(in_conv)))
        assert in_conv.shape == in_skip.shape
        in_ = torch.cat((in_conv,in_skip),1)
        in_ = self.drop(self.conv1(self.pad(in_)))
        in_ = self.actv(self.drop(self.conv2(self.pad(in_))))
        return in_
    
class outconv(nn.Module):
    def __init__(self,in_c,out_c,drop=0.6):
        super(outconv,self).__init__()
        self.actv = nn.Softmax(-1)
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear',align_corners=False)
        self.pad = nn.ConstantPad3d((1,1,1,1,0,0),0)
        
        self.out1 = nn.Conv2d(in_channels=in_c, out_channels=in_c,
                 kernel_size=3)
        self.out2 = nn.Conv2d(in_channels=in_c, out_channels=out_c,
                 kernel_size=3)
        init(self.out1.weight)
        init(self.out2.weight)
        self.drop = nn.Dropout(p=drop)
        
    def forward(self,x):
        x = self.out1(self.pad(self.up(x)))
        x = self.out2(self.pad(self.drop(x)))
        # return self.actv(x)
        return x
