import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from resnet import conv_block_preact, BottleneckResidualUnitPreact


class FPN_ResNetPreact(nn.Module):
    def __init__(self, block, num_blocks, init_channels=3):
        super().__init__()
        self.init_channels = init_channels
        self.in_channels = 64
        
        self.make_c1 = nn.Sequential(
            nn.Conv2d(self.init_channels, self.in_channels, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )
        self.make_c2 = self._make_layer(block, num_blocks[0], out_channels=64, stride=1)  # Final out_channels is (out_channels * block.expansion)
        self.make_c3 = self._make_layer(block, num_blocks[1], out_channels=128, stride=2)
        self.make_c4 = self._make_layer(block, num_blocks[2], out_channels=256, stride=2)
        self.make_c5 = self._make_layer(block, num_blocks[3], out_channels=512, stride=2)
        self.c5_conv1x1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # (== make_p5)
        
        # Lateral layers
        self.lateral_c2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        
        self.lateral_c3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.lateral_c4 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        
        # Smooth p_n for Anti-aliasing
        self.smooth_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_blocks-1)  # ex) (num_blocks == 3 and stride == 2) --> [2, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _upsample_element_wise_add(self, top, bottom):
        """
        Upsample [top] and element-wise add [top] to [bottom]
        """
        _, _, h, w = bottom.shape
        return F.upsample(top, size=(h, w), mode='bilinear') + bottom
    
    def forward(self, x):
        # Bottom-up
        c1 = self.make_c1(x)
        c2 = self.make_c2(c1)
        c3 = self.make_c3(c2)
        c4 = self.make_c4(c3)
        c5 = self.make_c5(c4)
        
        # Top-down
        p5 = self.c5_conv1x1(c5)
        p4 = self._upsample_element_wise_add(top=p5, bottom=self.lateral_c4(c4))
        p3 = self._upsample_element_wise_add(top=p4, bottom=self.lateral_c3(c3))
        p2 = self._upsample_element_wise_add(top=p3, bottom=self.lateral_c2(c2))
        
        # Smooth
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)
        
        return p2, p3, p4, p5


def FPN_ResNet50Preact():
    return FPN_ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 4, 6, 3])

def FPN_ResNet101Preact():
    return FPN_ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 4, 23, 3])

def FPN_ResNet152Preact():
    return FPN_ResNetPreact(block=BottleneckResidualUnitPreact, num_blocks=[3, 8, 36, 3])

def test_output_shape(model):
    input_tensor = torch.randn(1, 3, 224, 224)
    p2, p3, p4, p5 = model(input_tensor)
    print()
    print('=' * 60)
    print(f' * input shape: {input_tensor.shape}')
    print(f' * p2 shape: {p2.shape}')
    print(f' * p3 shape: {p3.shape}')
    print(f' * p4 shape: {p4.shape}')
    print(f' * p5 shape: {p5.shape}')
    print('=' * 60)
    print()

if __name__ == '__main__':
    model = FPN_ResNet50Preact()
    test_output_shape(model=model)
    summary(model=model.cpu(), input_size=(3, 224, 224), device='cpu')
