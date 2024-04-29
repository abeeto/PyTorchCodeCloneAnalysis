import torch
import torch.nn as nn
import torch.nn.init as init

from network_module import *

def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    print('Initialize network with %s type' % init_type)
    net.apply(init_func)

###========================== Diverse types of ResNet framework ==========================
# Fully convolutional layers in feature extraction part compared to Gray_VGG16_BN
# This is for generator only, so BN cannot be attached to the input and output layers of feature extraction part
# Each output of block (conv*) is "convolutional layer + LeakyReLU" that avoids feature sparse
# We replace the adaptive average pooling layer with a convolutional layer with stride = 2, to ensure the size of feature maps fit classifier
class BasicBlock_BN(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, first = False):
        super(BasicBlock_BN, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = 4
        self.stride = stride
        self.first = first
        self.conv1 = Conv2dLayer(inplanes, planes, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.conv2 = Conv2dLayer(planes, planes, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        if stride != 1 or first == True:
            self.downsample = Conv2dLayer(inplanes, planes * self.expansion, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.stride != 1 or self.first == True:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
       
class BasicBlock_IN(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, first = False):
        super(BasicBlock_IN, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = 4
        self.stride = stride
        self.first = first
        self.conv1 = Conv2dLayer(inplanes, planes, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        self.conv2 = Conv2dLayer(planes, planes, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        if stride != 1 or first == True:
            self.downsample = Conv2dLayer(inplanes, planes * self.expansion, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.stride != 1 or self.first == True:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class Bottleneck_BN(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, first = False):
        super(Bottleneck_BN, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = 4
        self.stride = stride
        self.first = first
        self.conv1 = Conv2dLayer(inplanes, planes, 1, 1, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.conv2 = Conv2dLayer(planes, planes, 3, stride, 1, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.conv3 = Conv2dLayer(planes, planes * self.expansion, 1, 1, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        if stride != 1 or first == True:
            self.downsample = Conv2dLayer(inplanes, planes * self.expansion, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride != 1 or self.first == True:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
        
class Bottleneck_IN(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, first = False):
        super(Bottleneck_IN, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = 4
        self.stride = stride
        self.first = first
        self.conv1 = Conv2dLayer(inplanes, planes, 1, 1, 0, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        self.conv2 = Conv2dLayer(planes, planes, 3, stride, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        self.conv3 = Conv2dLayer(planes, planes * self.expansion, 1, 1, 0, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        if stride != 1 or first == True:
            self.downsample = Conv2dLayer(inplanes, planes * self.expansion, 1, stride, 0, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride != 1 or self.first == True:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet_BN(nn.Module):
    def __init__(self, block, layers, in_channels = 3, num_classes = 1000):
        super(ResNet_BN, self).__init__()
        self.inplanes = 64
        self.begin1 = Conv2dLayer(in_channels, 64, 7, 1, 3, pad_type = 'zero', activation = 'lrelu', norm = 'none')
        self.begin2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.begin3 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'bn')
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.fusion = Conv2dLayer(2048, 512, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'none')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        layers = []
        first = block(self.inplanes, planes, stride = stride, first = True)
        layers.append(first)
        self.inplanes = planes * first.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride = 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.begin1(x)
        x = self.begin2(x)
        x = self.begin3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fusion(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet_IN(nn.Module):
    def __init__(self, block, layers, in_channels = 3, num_classes = 1000):
        super(ResNet_IN, self).__init__()
        self.inplanes = 64
        self.begin1 = Conv2dLayer(in_channels, 64, 7, 1, 3, pad_type = 'zero', activation = 'lrelu', norm = 'none')
        self.begin2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        self.begin3 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'zero', activation = 'lrelu', norm = 'in')
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.fusion = Conv2dLayer(2048, 512, 3, 1, 1, pad_type = 'zero', activation = 'lrelu', norm = 'none')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        layers = []
        first = block(self.inplanes, planes, stride = stride, first = True)
        layers.append(first)
        self.inplanes = planes * first.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride = 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.begin1(x)
        x = self.begin2(x)
        x = self.begin3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fusion(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    model = ResNet_BN(Bottleneck_BN, [3, 4, 3, 3]).cuda()
    #model = Bottleneck_IN(64, 64, 1, first = True).cuda()
    A = torch.randn(1, 3, 224, 224).cuda()
    c = torch.ones(1, 1).long().cuda()
    b = model(A)
    print(b.shape)
    print(c.shape)
    '''
    criterion = torch.nn.CrossEntropyLoss().cuda()
    loss = criterion(b, c)
    '''
    loss = torch.mean(b)
    loss.backward()
