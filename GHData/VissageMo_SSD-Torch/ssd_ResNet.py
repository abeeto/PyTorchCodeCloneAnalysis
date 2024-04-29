import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from layers import *
from data import voc, coco


class SSDResNet(nn.Module):
    """
    SSD model with ResNet50 as backbone.

    Args:
        phase: 'test' for testing , 'train' for training
        size: 
        base: backbone that used in 

    Return:
    """

    def __init__(self, phase, num_classes):
        super(SSD, self).__init__()
        # SSD Init
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.num_classes = num_classes

        # ResNet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Extra layer
        self.extra = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), # 3x3
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(128, 256, 3, stride=2) # 1x1
        )

        # Localization layer
        self.loc = nn.Sequential(
            nn.Conv2d(, 4 * 4, 3, padding=1),
            nn.Conv2d(, 6 * 4, 3, padding=1),
            nn.Conv2d(, 6 * 4, 3, padding=1), # 7x7
            nn.Conv2d(, 6 * 4, 3, padding=1), # 5x5
            nn.Conv2d(1024, 4 * 4, 3, padding=1), # 3x3
            nn.Conv2d(256, 4 * 4, 3, padding=1) # 1x1
        )

        # Param init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Applies network layers and ops on input image x.

        Args:
            x: input image or batch of images. Shape: [batch, 3, 224, 224]
        
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
        """

        sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4

        x = self.layer1(x) # 1/8
        sources.append(x)
        x = self.layer2(x) # 1/16
        sources.append(x)
        x = self.layer3(x) # 1/32
        sources.append(x)
        x = self.layer4(x) # 1/64
        sources.append(x)

        for k, v in enumerate(self.extra):
            x = F.relu(v(x), input=True)
            if k % 2 == 1:
                sources.append(x) # 1/128, 1/256

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # flatten loc and conf to one line 
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1 for o in conf)], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext == '.pkl' or 'pth':
            print('Loading weights into model')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Only .pth and .pkl files supported.')


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

