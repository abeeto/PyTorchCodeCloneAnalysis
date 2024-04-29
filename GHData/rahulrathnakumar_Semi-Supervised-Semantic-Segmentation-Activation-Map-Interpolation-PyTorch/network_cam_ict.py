import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet
import torch.nn.functional as F

import GPUtil


cfg = {
    'modelf' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'modelg1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'modelg1ext': [512, 512, 512, 'M', 512, 512, 512, 'M'],
    'decoder': ['r', 512, 256, 128, 64, 32]
}

ranges = {
    'modelf': ((0,5), (5,10), (10,17),(17,24), (24,31)),
    'modelg1': ((0,5), (5,10), (10,17)),
    'modelg1_ext': ((0,7), (7,14))
}

class VGGNet(VGG):
    def __init__(self, n_class, pretrained = True, model = 'modelf', requires_grad = True, remove_fc = True, show_params = False):
        super().__init__(make_layers(cfg[model], batch_norm=False))
        self.ranges = ranges[model]
        self.n_class = n_class
        if pretrained:
            self.load_state_dict(models.vgg16(pretrained=True).state_dict())
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False
        if remove_fc:
            del self.classifier
            del self.avgpool
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.conv1x1 = nn.ModuleDict(
                {'x5': nn.Conv2d(512, self.n_class, 1, bias = False),
                'x4': nn.Conv2d(512, self.n_class, 1, bias = False),
                'x3': nn.Conv2d(256, self.n_class, 1, bias = False),
                'x2': nn.Conv2d(128, self.n_class, 1, bias = False),
                'x1': nn.Conv2d(64, self.n_class, 1, bias = False)
                }
            )
        if show_params:
            for name,param in self.named_parameters():
                print(name, param.size())
    def compute_cam(self, x, layer):
        x_ = x.clone()
        x_ = self.avgpool(x_)
        x_ = self.conv1x1[layer](x_)
        cam = F.conv2d(x, self.conv1x1[layer].weight)
        cam = F.softmax(cam, dim = 1)
        return cam
    def forward(self, x, cams = True):
        output = []
        input_size_h = x.size()[2]
        input_size_w = x.size()[3]
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)                
            output.append(x)
        if cams:
            x5_cam = self.compute_cam(output[4], 'x5')
            x4_cam = self.compute_cam(output[3], 'x4')
            x3_cam = self.compute_cam(output[2], 'x3')
            x2_cam = self.compute_cam(output[1], 'x2')
            x1_cam = self.compute_cam(output[0], 'x1')
            cam = [x1_cam, x2_cam, x3_cam, x4_cam, x5_cam]
            return cam, output
        else:
            return output



class network1(nn.Module):
    '''
    Network model 'f'
    '''
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1x1_mix = nn.ModuleDict(
                {'x5': nn.Conv2d(self.n_class, 512, 1, bias = False),
                'x4': nn.Conv2d(self.n_class, 512,  1, bias = False),
                'x3': nn.Conv2d(self.n_class, 256, 1, bias = False),
                'x2': nn.Conv2d(self.n_class, 128, 1, bias = False),
                'x1': nn.Conv2d(self.n_class, 64, 1, bias = False)
                }
        )
        self.decoder = decoder(cfg['decoder'], n_class, in_channels= 512)

    def compute_feature(self, x, layer):
        return self.conv1x1_mix[layer](x)
    def forward(self,output, cam_in = False):
        if cam_in:
            x5 = self.compute_feature(output[4], 'x5')
            x4 = self.compute_feature(output[3], 'x4')
            x3 = self.compute_feature(output[2], 'x3')
            x2 = self.compute_feature(output[1], 'x2')
            x1 = self.compute_feature(output[0], 'x1')
        else:
            x5 = output[4]  # size=(N, 512, x.H/32, x.W/32)
            x4 = output[3]  # size=(N, 512, x.H/16, x.W/16)
            x3 = output[2]  # size=(N, 256, x.H/8,  x.W/8)
            x2 = output[1]  # size=(N, 128, x.H/4,  x.W/4)
            x1 = output[0]  # size=(N, 64, x.H/2,  x.W/2)
        mask = self.decoder[2](self.decoder[0](self.decoder[1](x5)))
        mask = mask + x4
        mask = self.decoder[4](self.decoder[0](self.decoder[3](mask)))
        mask = mask + x3
        mask = self.decoder[6](self.decoder[0](self.decoder[5](mask)))
        mask = mask + x2
        mask = self.decoder[8](self.decoder[0](self.decoder[7](mask)))
        mask = mask + x1
        mask = self.decoder[10](self.decoder[0](self.decoder[9](mask)))
        mask = self.decoder[11](mask)
        return mask


def decoder(cfg, n_class, in_channels, batch_norm = True):
    layers = []
    for v in cfg:
        if v == 'r':
            layers += [nn.ReLU(inplace=True)]
        else:
            layers += [nn.ConvTranspose2d(in_channels, v, kernel_size=3, stride = 2, padding = 1, dilation = 1, output_padding= 1)]
            in_channels = v
            if batch_norm == True:
                layers += [nn.BatchNorm2d(v)]
    layers += [nn.Conv2d(32, n_class, kernel_size=1)]
    return nn.Sequential(*layers)


def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

