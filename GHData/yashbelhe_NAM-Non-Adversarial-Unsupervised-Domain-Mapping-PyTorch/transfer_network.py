import os, sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

class Net(nn.Module):
    def __init__(self, config):

        super(Net, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        dim_t      = config['input_dim_t']
        dim_s      = config['input_dim_s']
        depths     = config['crn_depths']
        ls         = config['lrelu_slope']
        img_size   = config['img_size']

        self.start_size = config['start_size']
        self.num_layers = len(depths)

        assert self.start_size*(2**(self.num_layers)) == img_size, "start_size*(2**(self.num_layers)) != dim_s"

        conv = []
        bn = []
        lrelu = []

        in_depth = dim_t
        for i in range(self.num_layers):
            conv.append(nn.Conv2d(in_depth, depths[i], (3, 3), (1, 1), (1, 1), bias=False))
            bn.append(nn.BatchNorm2d(depths[i]))
            lrelu.append(nn.LeakyReLU(ls))
            in_depth = depths[i] + dim_t
        
        self.conv_module  = nn.ModuleList(conv)
        self.bn_module    = nn.ModuleList(bn)
        self.lrelu_module = nn.ModuleList(lrelu)
        
        self.final_conv = nn.Conv2d(in_depth, dim_s, (3, 3), (1, 1), (1, 1))

    def forward(self, downsampled_images):

        x = downsampled_images[0]

        for i in range(self.num_layers):
            x = self.conv_module[i](x)
            x = self.lrelu_module[i](self.bn_module[i](x))
            x = self.upsample(x)
            x = torch.cat([x, downsampled_images[i + 1]], 1)

        x = self.final_conv(x)

        return x
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def get_downsampled_images(self, image):
        downsampled_images = []
        for i in range(self.num_layers + 1):
            s = self.start_size*(2**i)
            downsampled_images.append(F.upsample(image, size=(s,s), mode='bilinear'))
        return downsampled_images