import torch
import torch.nn as nn
from torchsummary import summary
from SKNET import SKConv
from SKNET import SKConv1x1

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock2(nn.Module):
    expansion: int = 4
    
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        norm_layer = None):
        
        super(BasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class SKBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        norm_layer = None):
        
        super(SKBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SKConv(planes, G = 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class SKBlock2(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        norm_layer = None):
        
        super(SKBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SKConv(planes, G = 1)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class SKBlock1x1(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        norm_layer = None):
        
        super(SKBlock1x1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SKConv1x1(planes, G = 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):
    
    def __init__(self,num_classes, layers, block = BasicBlock2): 
        
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block

        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    
    def _make_layer(self, planes, blocks, stride = 1):
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self.block.expansion, stride), 
                norm_layer(planes * self.block.expansion))

        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet18(num_classes, skconv = False, use1x1 = False):
    
    if skconv:
        if use1x1:
            return ResNet(num_classes, [2, 2, 2, 2], SKBlock1x1)
            
        return ResNet(num_classes, [2, 2, 2, 2], SKBlock2)
    
    return ResNet(num_classes, [2, 2, 2, 2])

def resnet34(num_classes, skconv = False, use1x1 = False):
    
    if skconv:
        if use1x1:
            return ResNet(num_classes, [3, 4, 6, 3], SKBlock1x1)
        return ResNet(num_classes, [3, 4, 6, 3], SKBlock2)
    
    return ResNet(num_classes, [3, 4, 6, 3])
    
    
    
if __name__ == '__main__':
    net = resnet18(200, True).cuda()
    # print(summary(net, (3, 64, 64)))
    print(summary(net, (3, 56, 56)))
    torch.cuda.empty_cache()
    # c = SKConv(128)
    # x = torch.zeros(8,128,2,2)
    # print(c(x).shape)   