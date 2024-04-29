import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as fun

class Residual_Block(nn.Module):
    """
    残差网络中的块
    """
    def __init__(self,i_channel,o_channel):
        super(Residual_Block, self).__init__()
        self.conv1=nn.Conv3d(in_channels=i_channel,out_channels=o_channel,kernel_size=(2,2,2))
        self.bn1=nn.BatchNorm3d(o_channel)

        self.conv2=nn.Conv3d(in_channels=o_channel,out_channels=o_channel,kernel_size=(2,2,2))
        self.bn2=nn.BatchNorm3d(o_channel)

    def forward(self,x):

        out=self.conv1(x)
        out=self.bn1(out)
        out=fun.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=x
        return fun.relu(out)