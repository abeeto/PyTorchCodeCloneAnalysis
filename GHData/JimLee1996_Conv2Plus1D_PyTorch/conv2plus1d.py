import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class FactorizedConv3d(nn.Module):
    r"""Factorized 3D convolutional filter

    The input is composed of several planes with distinct spatial and time axes, 
    by performing a 2D convolution over the spatial axes to an intermediate subspace, 
    followed by a 1D convolution over the time axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolution kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        factor_rate (float): Scale the number of parameters in FactorizedConv3d. Default: 1.0
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        factor_rate=1.0
    ):
        super(FactorizedConv3d, self).__init__()

        stride = _triple(stride)
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)

        # Decompose the parameters into spatial and temporal components
        # by masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid aberrant
        # behavior such as padding being added twice.

        # the original kernel size t x d x d
        # spatial filter: kernel size 1 x d x d
        # temporal filter: kernel size t x 1 x 1

        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # Compute M_i in section 3.5. Note: a factor has
        intermediate_channels = math.floor(
            factor_rate * (
                kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels *
                out_channels
            ) / (
                kernel_size[1] * kernel_size[2] * in_channels +
                kernel_size[0] * out_channels
            )
        )

        # The spatial conv is effectively a 2D conv, followed by BatchNorm and ReLU.
        self.spatial_conv = nn.Conv3d(
            in_channels,
            intermediate_channels,
            kernel_size=spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            bias=bias
        )
        self.bn = nn.BatchNorm3d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)

        # The temporal conv is effectively a 1D conv, NOT followed by BatchNorm and ReLU,
        # to make this module externally identical to a standard Conv3D for code reuse.
        self.temporal_conv = nn.Conv3d(
            intermediate_channels,
            out_channels,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=bias
        )

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

    def param_init(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


# test code
if __name__ == '__main__':

    from torchsummary import summary

    size = (3, 16, 112, 112)

    model = FactorizedConv3d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        factor_rate=0.5
    )

    summary(model, size)
