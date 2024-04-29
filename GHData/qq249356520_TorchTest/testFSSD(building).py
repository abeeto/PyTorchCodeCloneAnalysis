# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

extras = {
    '300': [256, 512, 128, 'S', 256],
    '512': [256, 512, 128, 'S', 256],
}

fea_channels = {
    '300': [512, 512, 256, 256, 256, 256],
    '512': [512, 512, 256, 256, 256, 256, 256]}

size = 300

#调用：vgg(vgg_base[str(size)], 3)
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i  #输入图片是三通道的
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
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

def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

for layer in vgg(vgg_base[str(size)], 3):
    print(layer)

print("-----------------extra--------------------")
for layer in add_extras(extras[str(size)], 1024):
    print(layer)

def feature_transform_module(vgg, extral, size):
    if size == 300:
        up_size = 38
    elif size == 512:
        up_size = 64

    layers = []
    #conv4_3
    layers += [BasicConv(vgg[24].out_channels, 256, kernel_size=1, padding=0)]
    #fc_7
    layers += [BasicConv(vgg[-2].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    #feature 1
    layers += [BasicConv(extral[-1].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]

    return vgg, extral, layers

def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)

def pyramid_feature_extractor(size):
    layers = []
    if size == 300:
        layers = [#级联concat 38 ×38 × 768-->> 38 × 38 × 512
                  BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  #19
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1),
                  #10
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  #5
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  #3
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  #1
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1)]
        return layers

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, \
                 stride=1, padding=0, dilation=1, groups=1, relu=True,\
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class FSSD(nn.Module):
    """
    Args:
        base: VGG16 layer for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base, extras, ft_module, pyramid_ext, head, num_classes, size):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        #TODO:
        self.size = size

        #SSD
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.ft_module = nn.ModuleList(ft_module) #需要级联的层
        self.pyramid_ext = nn.ModuleList(pyramid_ext)
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x

        Args:
            :param x: input image or batch of images. shape[batch, batch * 3, 300, 300]
            :param test: phase
        :return:
            Depanding on phase:
            test:
                Variable(tensor) of output class label predictions
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch, topk, 7]
            train:
                list of concat outputs from::
                    1: confidence layers, Shape: [batch * num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors * 4]
                    3: priorbox layers, Shape: [2, num_priors * 4]
        """
        source_features = list()
        transformed_teatures = list()
        loc = list()
        conf = list()

        #conv4_3之前  vgg[24]是conv4_3
        #apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)
        #conv4_3
        source_features.append(x)

        for k in range(23, len(self.base)):
            x = self.base[k](x)
        # fc7
        source_features.append(x)

        #apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
        source_features.append(x)
        assert len(self.ft_module) == len(source_features) #需要级联
        for k, v in enumerate(self.ft_module):
            transformed_teatures.append(v(source_features[k])) #级联之前的卷积 三个特征曾分别输入变量x进行卷积操作
        concat_fea = torch.cat(transformed_teatures, 1)  #channels
        x = self.fea_bn(concat_fea)
        pyramid_fea = list()

        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x) #完成级联后的新的金字塔特征

        #apply multibox head to source layers
        for(x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  #contiguous后可以view 指向临近指针 否则报错
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  #view 将数组变为按照-1维度的数组， -1按照原参数的列数自行推断
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4), #loc preds
                self.softmax(conf.view(-1, self.num_classes))) #conf preds
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))
        return output


#调用：add_extras(extras[str(size)], 1024)
