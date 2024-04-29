from torch import nn
import torch.nn.functional as F
import numpy as np

class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, activation,
                 kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.dilation_block = nn.Sequential(nn.BatchNorm3d(in_channels),
                                        activation,
                                        nn.Conv3d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  padding=dilation,
                                                  stride=stride,
                                                  dilation=dilation,
                                                  bias=bias))
    def forward(self, x):
        x = self.dilation_block(x)
        return x

class HighResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, activation,
                 kernel_size=3, stride=1, bias=False, n_conv_blocks=2):
        super().__init__()
        # concatenating n_conv_blocks
        highresnet_block = nn.ModuleList()
        for _ in range(n_conv_blocks):
            dilation_block = DilationBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        dilation=dilation,
                                        activation=activation,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        bias=bias)
            highresnet_block.append(dilation_block)
            in_channels = out_channels
        self.highresnet_block = nn.Sequential(*highresnet_block)
    # Fixed of by zero-padding in element op because
    # n_param_flow is always greater than n_bypass_flow
    def element_op(self, out, x):
        n_param_flow = out.shape[1]
        n_bypass_flow = x.shape[1]
        pad_1 = np.int((n_param_flow - n_bypass_flow) // 2)
        pad_2 = np.int(n_param_flow - n_bypass_flow - pad_1)
        padding_dims = [0, 0] * 3 + [pad_1, pad_2]
        # padding channels dimension
        x = F.pad(x, padding_dims, "constant", 0)
        return x + out
    def forward(self, x):
        out = self.highresnet_block(x)
        x = self.element_op(out, x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, padding,
                 kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias)
        self.bnorm = nn.BatchNorm3d(out_channels)
        self.activation = activation
    def forward(self, x):
        x =self.conv(x)
        x =self.bnorm(x)
        if self.activation == None:
            return x
        else:
            x =self.activation(x)
            return x

class HighRes3DNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation = nn.PReLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_out_channels = 16
        self.last_last_conv_channels = 80
        self.n_dilation = 3
        self.n_highresnet_blocks = 3
        self.bias = False
        self.kernel_size = 3
        self.stride = 1
        self.first_padding = 1
        self.activation = activation
        # first convolutional block
        self.first_conv_block = ConvBlock(in_channels = self.in_channels,
                                          out_channels = self.first_out_channels,
                                          activation = self.activation,
                                          padding=self.first_padding)
        # highresnet and dilation blocks
        blocks = nn.ModuleList()
        for ii in range(self.n_dilation):
            for jj in range(self.n_highresnet_blocks):
                dilation = 2**ii
                if ii == 0:
                    in_channel = self.first_out_channels
                    out_channel = self.first_out_channels
                elif ii > 0 and jj == 0:
                    in_channel = self.first_out_channels * (2 ** (ii-1))
                    out_channel = self.first_out_channels * dilation
                else:
                    in_channel = self.first_out_channels * dilation
                    out_channel = self.first_out_channels * dilation
                block = HighResNetBlock(in_channel,
                                out_channel,
                                dilation, self.activation)
                blocks.append(block)
        self.highresnet_blocks = nn.Sequential(*blocks)
        # convolutional block
        self.out_conv_blocks = \
            nn.Sequential(ConvBlock(in_channels=out_channel,
                                    out_channels=self.last_last_conv_channels,
                                    activation=self.activation,
                                    padding=0,
                                    kernel_size=1),
                          ConvBlock(in_channels=self.last_last_conv_channels,
                                    out_channels=self.out_channels,
                                    activation=None,
                                    padding=0,
                                    kernel_size=1))
    def forward(self, x):
        x = self.first_conv_block(x)
        x = self.highresnet_blocks(x)
        x = self.out_conv_blocks(x)
        return x


