#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

base = {
    '300' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
             512, 512, 512]
}

def vgg(cfg, picture_channels, batch_norm=False):
    layers = []
    in_channels = picture_channels
    for v in cfg:
        if v == 'M': #Maxpooling 并且不进行边缘修补
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C': #边缘补nan
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:#卷积前后维度可以通过字典中数据设置好
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

vgg_model = vgg(base[str(300)], 3)


extras = {
    '300' : [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
}
def add_extra(cfg, base_channel, batch_norm=False):
    layers = []
    in_channels = base_channel
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S' : #s代表stride 为2时就相当于缩小feature map
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

exter_layer = add_extra(extras[str(300)], 1024)

print('-----------------extra-------------------')
for i in exter_layer:
    print(i)

mbox = {
    '300' : [4, 6, 6, 6, 4, 4]
}

def multibox(vgg, extra_layers, cfg, num_class):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2] #第21层和倒数第二层
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_class, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  cfg[k] * num_class, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)

base_, extras_, head_ = multibox(vgg_model, exter_layer, mbox[str(300)], 21)

print('--------------------------vgg---------------------')
for i in base_:
    print(i)
print('-----------------extra-------------------')
for i in extras_:
    print(i)
for i, j in enumerate(head_):
    if i == 0:
        print("--------predction : loc----------")
    else:
        print("--------predction : conf----------")

    for k in j:
        print(k)


class SSD(nn.Module):
    """
    Single Shot Multibox Architecture

    Args:
        phase: 'train' or 'test'
        size: input image size
        base: vgg16
        extras: ssd layer
        head: loc and conf layer
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes

        #SSD network
        self.vgg = nn.ModuleList(base)
        # self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        """
        Applies network layer and ops on input image(s) x

        Args:
        param x: input image or batch of images shape[batch, 3, 300, 300]
        :return:
        """

        sources = list()
        loc = list()
        conf = list()

        #apply vgg up to conv4_3 relu

        for k in range(23):
            x = self.vgg[k](x) #得到[1, 512, 38, 38]

        #s = self.L2Norm(x)
        sources.append(x)

        #apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x) #得到[1, 1024, 19, 19]
        sources.append(x)

        #apply extra layers and cache source layer oputputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if(k % 2 == 1):
                sources.append(x)
            """
            得到剩下的
            torch.Size([1, 512, 10, 10])
            torch.Size([1, 256, 5, 5])
            torch.Size([1, 256, 3, 3])
            torch.Size([1, 256, 1, 1])
            """
        #apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


