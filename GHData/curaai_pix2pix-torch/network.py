import torch
import torchvision
import torch.nn as nn 
import numpy as np


def conv_block(idx, name, in_c, out_c, activation, kernel_size=3, stride=1, padding=1, transpose=False, bn=True, bias=True, drop=False):
    block = nn.Sequential()

    if not transpose:
        block.add_module(name + ' Conv2d' + idx, nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    else:
        block.add_module(name + ' Conv2d_Transpose' + idx, nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    if bn:
        block.add_module(name + ' Batch_norm' + idx, nn.BatchNorm2d(out_c))
    if activation == 'relu':
        block.add_module(name + ' ReLU' + idx, nn.ELU(inplace=True))
    elif activation == 'leaky_relu':
        block.add_module(name + ' Leaky_ReLU' + idx, nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'sigmoid':
        block.add_module(name + ' Sigmoid' + idx, nn.Sigmoid())
    elif activation == 'tanh':
        block.add_module(name + ' Tanh' + idx, nn.Tanh())
    if drop:
        block.add_module(name + " Drop_out" + idx, nn.Dropout())
    
    return block


class G(nn.Module):
    """
    input : ? x 128 x 128 x 3
    layer0 : ? x 128 x 128 x 32
    layer1 : ? x 64 x 64 x 64
    layer2 : ? x 32 x 32 x 128
    layer3 : ? x 16 x 16 x 256
    layer4 : ? x 8 x 8 x 512
    layer5 : ? x 4 x 4 x 1024

    dlayer4 : ? x 8 x 8 x 512
    dlayer3 : ? x 16 x 16 x 256
    dlayer2 : ? x 32 x 32 x 128
    dlayer1 : ? x 64 x 64 x 64
    dlayer0 : ? x 128 x 128 x 33
    """
    def __init__(self):
        super(G, self).__init__()
        self.name = "G"

        self.build()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def build(self):
        activation = 'leaky_relu'
        self.pooling = nn.Sequential(nn.AvgPool2d((2, 2), 2))

        # down
        self.layer0_0 = conv_block('0_0', self.name, 3, 32, activation, kernel_size=4, stride=2)
        self.layer0_1 = conv_block('0_1', self.name, 32, 32, activation)

        self.layer1_0 = conv_block('1_0', self.name, 32, 64, activation, kernel_size=4, stride=2)
        self.layer1_1 = conv_block('1_1', self.name, 64, 64, activation)

        self.layer2_0 = conv_block('2_0', self.name, 64, 128, activation, kernel_size=4, stride=2)
        self.layer2_1 = conv_block('2_1', self.name, 128, 128, activation)

        self.layer3_0 = conv_block('3_0', self.name, 128, 256, activation, kernel_size=4, stride=2) 
        self.layer3_2 = conv_block('3_2', self.name, 256, 256, activation, drop=True) 
        
        self.layer4_0 = conv_block('4_0', self.name, 256, 512, activation, kernel_size=4, stride=2)
        self.layer4_2 = conv_block('4_2', self.name, 512, 256, activation)

        # up
        self.dlayer4_0 = conv_block('up4_0', self.name, 512, 512, activation, transpose=True, kernel_size=4, stride=2)
        self.dlayer4_2 = conv_block('up4_2', self.name, 512, 256, activation, drop=True)

        self.dlayer3_0 = conv_block('up3_0', self.name, 512, 256, activation, transpose=True, kernel_size=4, stride=2)
        self.dlayer3_2 = conv_block('up3_2', self.name, 256, 128, activation)

        self.dlayer2_0 = conv_block('up2_0', self.name, 256, 128, activation, transpose=True, kernel_size=4, stride=2)
        self.dlayer2_1 = conv_block('up2_1', self.name, 128, 64, activation)

        self.dlayer1_0 = conv_block('up1_0', self.name, 128, 64, activation, transpose=True, kernel_size=4, stride=2)
        self.dlayer1_1 = conv_block('up1_1', self.name, 64, 32, activation)

        self.dlayer0_0 = conv_block('up0_0', self.name, 64, 32, activation, transpose=True, kernel_size=4, stride=2)
        self.dlayer0_1 = conv_block('up0_1', self.name, 32, 3, 'tanh')

    def forward(self, x):
        out0_0 = self.layer0_0(x)
        out0_1 = self.layer0_1(out0_0)
        
        out1_0 = self.layer1_0(out0_1)
        out1_1 = self.layer1_1(out1_0)

        out2_0 = self.layer2_0(out1_1)
        out2_1 = self.layer2_1(out2_0)

        out3_0 = self.layer3_0(out2_1)
        out3_2 = self.layer3_2(out3_0)

        out4_0 = self.layer4_0(out3_2)
        out4_2 = self.layer4_2(out4_0)
        
        cat_out5_2 = torch.cat((out4_2, self.pooling(out3_2)), 1)
        dout4_0 = self.dlayer4_0(cat_out5_2)
        dout4_2 = self.dlayer4_2(dout4_0)

        cat_out4_2 = torch.cat((dout4_2, out3_2), 1)
        dout3_0 = self.dlayer3_0(cat_out4_2)
        dout3_2 = self.dlayer3_2(dout3_0)

        cat_out3_2 = torch.cat((dout3_2, out2_1), 1)
        dout2_0 = self.dlayer2_0(cat_out3_2)
        dout2_1 = self.dlayer2_1(dout2_0)

        cat_out2_1 = torch.cat((dout2_1, out1_1), 1)
        dout1_0 = self.dlayer1_0(cat_out2_1)
        dout1_1 = self.dlayer1_1(dout1_0)

        cat_out1_1 = torch.cat((dout1_1, out0_1), 1)
        dout0_0 = self.dlayer0_0(cat_out1_1)
        dout0_1 = self.dlayer0_1(dout0_0)

        return dout0_1


class D(nn.Module):
    """
    input : ? x 128 x 128 x 6(3 x 2)
    layer0 : ? x 128 x 128 x 32 
    layer1 : ? x 64 x 64 x 64 
    layer2 : ? x 32 x 32 x 128 
    layer3 : ? x 31 x 31 x 256 
    output : ? x 30 x 30 x 1 
    """
    def __init__(self):
        super(D, self).__init__()
        self.name = "D"
        
        self.build()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def build(self):
        activation = 'leaky_relu'

        self.layer0 = conv_block('0', self.name, 6, 32, activation, bn=False, kernel_size=4, stride=2)

        self.layer1_0 = conv_block('1_0', self.name, 32, 64, activation, kernel_size=4, stride=2)
        self.layer1_1 = conv_block('1_1', self.name, 64, 64, activation)

        self.layer2_0 = conv_block('2_0', self.name, 64, 128, activation)
        self.layer2_1 = conv_block('2_1', self.name, 128, 128, activation)

        self.layer3_0 = conv_block('3_0', self.name, 128, 256, activation, kernel_size=2, stride=1, padding=0) 
        self.layer3_1 = conv_block('3_1', self.name, 256, 256, activation) 
        self.layer3_2 = conv_block('3_2', self.name, 256, 256, activation) 
        
        self.layer4_0 = conv_block('4_0', self.name, 256, 512, activation)
        self.layer4_1 = conv_block('4_1', self.name, 512, 512, activation)
        self.layer4_2 = conv_block('4_2', self.name, 512, 1, 'sigmoid', kernel_size=2, stride=1, padding=0)

    def forward(self, src_input, trg_input):
        x = torch.cat((src_input, trg_input), 1)
        out0 = self.layer0(x)

        out1_0 = self.layer1_0(out0)
        out1_1 = self.layer1_1(out1_0)

        out2_0 = self.layer2_0(out1_1)
        out2_1 = self.layer2_1(out2_0)

        out3_0 = self.layer3_0(out2_1)
        out3_1 = self.layer3_1(out3_0)
        out3_2 = self.layer3_2(out3_1)

        out4_0 = self.layer4_0(out3_2)
        out4_1 = self.layer4_1(out4_0)
        out4_2 = self.layer4_2(out4_1)

        return out4_2
