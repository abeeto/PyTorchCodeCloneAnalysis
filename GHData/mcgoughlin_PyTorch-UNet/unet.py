# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:47:26 2021

@author: mcgoug01
"""

import unet_modules as u
import torch
import torch.nn as nn
import numpy as np

class UNet(nn.Module):
    def __init__(self, depth:int=5, in_channels:int=32, out_labels:int=4):
        super(UNet,self).__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        in_ = in_channels
        self.in_conv = u.deconv(1, in_)
        for layer in range(depth+1):
            self.down_convs.append(u.deconv(in_, in_*2))
            self.up_convs.append(u.upconv(in_*2, in_))
            in_ *= 2
            
        self.out = u.outconv(in_channels,out_labels)
        
    
    def forward(self,x):
        x = torch.transpose(x,1,3)
        deconvs = []
        deconvs.append(self.in_conv(x))
        #compute encode path
        for i,encode in enumerate(self.down_convs):
            deconvs.append(encode(deconvs[i]))
        #compute bottom block in decode path
        x = self.up_convs[-1](deconvs.pop(-1),
                              deconvs.pop(-1))
        #compute rest of decode path
        for decode in range(len(self.up_convs)-2,-1,-1):
            x = self.up_convs[decode](x,deconvs.pop(-1))
        
        return self.out(x)
        
if __name__ == "__main__":
    unet = UNet(depth=4,in_channels=4,out_labels=4)
    a = torch.Tensor(np.arange(512*512).reshape(-1,512,512,1))
    c = unet(a)
    print(c[0])