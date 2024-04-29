import torch
import torch.nn as nn
from torchsummary import summary


def conv_block_preact(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
    )


class global_avg_pool2d(nn.Module):
    def forward(self, x):
        _, _, h, w = x.shape  # x.shape == (B, C, H, W)
        return nn.AvgPool2d(kernel_size=(h, w))(x)


class ResidualUnitPreact(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, channels, stride):
        super().__init__()
        self.conv_block_preact1 = conv_block_preact(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_block_preact2 = conv_block_preact(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if not (stride == 1 and in_channels == channels*self.expansion):  # not (Keep Feature Size and Keep number of filters)
            self.shortcut = conv_block_preact(in_channels, channels*self.expansion, kernel_size=1, stride=stride, bias=False)  # Projection Shortcut
            
    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        
        out = self.conv_block_preact1(x)
        out = self.conv_block_preact2(out)
        out += shortcut
        return out
    

class BottleneckResidualUnitPreact(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, channels, stride):
        super().__init__()
        self.conv_block_preact1 = conv_block_preact(in_channels, channels, kernel_size=1, bias=False)
        self.conv_block_preact2 = conv_block_preact(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_block_preact3 = conv_block_preact(channels, channels*self.expansion, kernel_size=1, bias=False)
        
        if not (stride == 1 and in_channels == channels*self.expansion):  # not (Keep Feature Size and Keep number of filters)
            self.shortcut = conv_block_preact(in_channels, channels*self.expansion, kernel_size=1, stride=stride, bias=False)  # Projection Shortcut
            
    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        
        out = self.conv_block_preact1(x)
        out = self.conv_block_preact2(out)
        out = self.conv_block_preact3(out)
        out += shortcut
        return out
    

class ResNetPreact(nn.Module):
    def __init__(self, block, num_blocks, num_classes, init_channels=3):
        super().__init__()
        self.init_channels = init_channels
        self.in_channels = 64
        
        self.init = nn.Sequential(
            nn.Conv2d(self.init_channels, self.in_channels, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )
        self.encoder = nn.Sequential(
            self._make_layer(block, num_blocks[0], out_channels=64, stride=1),  # Final out_channels is (out_channels * block.expansion)
            self._make_layer(block, num_blocks[1], out_channels=128, stride=2),
            self._make_layer(block, num_blocks[2], out_channels=256, stride=2),
            self._make_layer(block, num_blocks[3], out_channels=512, stride=2),
        )
        self.decoder = nn.Sequential(
            global_avg_pool2d(),
            nn.Flatten(),
            nn.Linear(512*block.expansion, num_classes),
            nn.Softmax(),
        )

    def _make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_blocks-1)  # ex) (num_blocks == 3 and stride == 2) --> [2, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.encoder(out)
        out = self.decoder(out)
        return out
    

def ResNet18Preact():  # (2 + 2 + 2 + 2) * 2 + 2 = 18
    return ResNetPreact(block=ResidualUnitPreact, num_blocks=[2, 2, 2, 2], num_classes=10)

def ResNet34Preact():  # (3 + 4 + 6 + 3) * 2 + 2 = 34
    return ResNetPreact(block=ResidualUnitPreact, num_blocks=[3, 4, 6, 3], num_classes=10)

def ResNet50Preact():  # (3 + 4 + 6 + 3) * 3 + 2 = 50
    return ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 4, 6, 3], num_classes=10)

def ResNet101Preact():  # (3 + 4 + 23 + 3) * 3 + 2 = 101
    return ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 4, 23, 3], num_classes=10)

def ResNet152Preact():  # (3 + 8 + 36 + 3) * 3 + 2 = 152
    return ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 8, 36, 3], num_classes=10)


if __name__ == '__main__':
    summary(model=ResNet50Preact().cpu(), input_size=(3, 224, 224), device='cpu')
