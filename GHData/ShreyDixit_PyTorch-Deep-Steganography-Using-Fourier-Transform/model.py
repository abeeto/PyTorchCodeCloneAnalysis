import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from collections import OrderedDict
from fastai.vision.models import DynamicUnet

class ReLUModified(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.clamp_min(0.)-0.5

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=conv3x3, activation = ReLUModified, *args, **kwargs):
        super().__init__()
        self.conv, self.in_channels, self.out_channels = conv,  in_channels, out_channels
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv),
            activation(),
            conv_bn(self.out_channels, self.out_channels, conv=self.conv),
        )
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            'bn' : nn.BatchNorm2d(self.out_channels)
            
        })) if self.should_apply_shortcut else None
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
    
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, activation = ReLUModified):
        super().__init__()
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, activation=activation),
            *[block(out_channels, out_channels, activation=activation) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    

class StegNet(nn.Module):
    def __init__(self, encoder_layers=5, decoder_layers=5):
        super().__init__()
        self.mean = torch.tensor([[[[[120.0647]], [[113.9922]], [[103.8980]]],
                                 [[[119.9911]], [[113.9572]], [[103.9110]]]]])
        self.std = torch.tensor([[[[[70.2821]], [[69.1352]], [[72.8606]]],
                                [[[70.2184]], [[69.1267]], [[72.9056]]]]])
        res = ResNetLayer(6, 32, n = encoder_layers)
        self.encoder = DynamicUnet(list(res.children())[0], 3, (128, 128))
        self.decoder = nn.Sequential(ResNetLayer(3, 32, n = decoder_layers), 
                                     nn.Conv2d(32, 3, kernel_size=(1, 1), bias=False),)
    
    def forward(self, X):
        X = (X-self.mean)/self.std
        concat_images = torch.cat((X[:, 0], X[:, 1]), 1)
        embedded_image = self.encoder(concat_images)
        decoded_image = self.decoder(embedded_image)
        out = torch.stack((embedded_image, decoded_image), 1)
        out = (out * self.std) + self.mean
        return out
