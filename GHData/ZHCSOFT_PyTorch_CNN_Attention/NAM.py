import torch
import torch.nn as nn


# reference: 2111.12419

class ChannelAttention(nn.Module):

    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.bn2 = nn.BatchNorm2d(in_planes, affine=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.sigmoid(x) * residual

        return x


class SpatialAttention(nn.Module):

    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()

        self.bn2 = nn.BatchNorm2d(in_planes, affine=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        batch, channel, height, width = x.size()
        residual = x

        x = x.permute(0, 2, 3, 1).reshape(batch, height*width, 1, channel).contiguous()
        x = self.bn2(x)

        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 3, 2, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.view(batch, channel, width, height).contiguous()
        x = self.sigmoid(x) * residual

        return x

if __name__ == '__main__':
    test_input = torch.rand(2, 3, 513, 513)
    test_out = SpatialAttention(513*513)(test_input)
    print('Spatial_out =', test_out)
    print('Spatial_out.size =', test_out.size())

    test_out = ChannelAttention(3)(test_input)
    print('Channel_out =', test_out)
    print('Channel_out.size =', test_out.size())
