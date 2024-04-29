import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['resnet50']


model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckNoBN(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BottleneckNoBN, self).__init__()
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetNoBN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None
                ):
        super(ResNetNoBN, self).__init__()
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
       

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetNoBN(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model


def resnet50NoBN(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', BottleneckNoBN, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class Net(nn.Module):
    def __init__(self,lastLayer=32):
        super(Net, self).__init__()
        resnet =  resnet50NoBN(pretrained=True)
        self.resNet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimRed = nn.Sequential(nn.Linear(2048,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),
                                    nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,lastLayer))
        self.classlyr = nn.Linear(lastLayer,2)
        
    def forward(self, x):
        x = self.resNet(x)
        x = torch.squeeze(x)
        x = F.relu(self.dimRed(x))
        return self.classlyr(x)
    

class Net_trip(nn.Module):
    def __init__(self):
        super(Net_trip, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet =  resnet50NoBN(pretrained=True)
        self.resNet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimRed = nn.Sequential(nn.Linear(2048,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),
                                    nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
    def forward(self, x):
        x = self.resNet(x)
        x = torch.squeeze(x)
        return self.dimRed(x)
    
# added on the 220614
class Net_trip_2(nn.Module):
    def __init__(self):
        super(Net_trip_2, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet =  resnet50NoBN(pretrained=True)
        self.resNet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimRed = nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Linear(512,128),
                                    nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,10))
    def forward(self, x):
        x = self.resNet(x)
        x = torch.squeeze(x)
        return self.dimRed(x)
    
