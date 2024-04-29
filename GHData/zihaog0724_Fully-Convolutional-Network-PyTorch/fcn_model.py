import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

pretrained_net = models.resnet34(pretrained=True)

class fcn_8x_resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(fcn_8x_resnet34, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.pad = nn.ZeroPad2d(padding=(0,1,1,0))

    def forward(self, x):
        x = self.stage1(x) 
        s1 = x # bs, 128, 63, 63
        s1 = self.pad(s1) # bs, 128, 64, 64

        x = self.stage2(x) 
        s2 = x  # bs, 256, 32, 32

        x = self.stage3(x)
        s3 = x # bs, 512, 16, 16

        s3 = self.scores1(s3) # bs, 2, 16, 16
        s3 = self.upsample_2x(s3) # bs, 2, 32, 32
        s2 = self.scores2(s2) # bs, 2, 32, 32
        s2 = s2 + s3 # bs, 2, 32, 32

        s1 = self.scores3(s1) # bs, 2, 64, 64
        s2 = self.upsample_4x(s2) # bs, 2, 64, 64
        s = s1 + s2 # bs, 2, 64, 64

        s = self.upsample_8x(s) # bs, 2, 512, 512
        
        return s




