import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import numpy as np
import torch.nn.functional as F

# x2
def conv_t_block_2x(in_ch, out_ch, dropout = 0.2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 3),
        nn.Dropout(dropout),
        
    )
# -4
def conv_block(in_ch, out_ch, dropout = 0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        
        dropout = 0.2
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)

        channels = (16, 32, 64, 128, 256, 512, 256, 128, 128, 64)

        arr = [
            nn.Conv2d(3, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            self.act_fn,
        ]


        for idx in range(0, len(channels) - 1):
            ch = channels[idx]
            next_ch = channels[idx + 1]
            arr = arr + [
                nn.Conv2d(ch, next_ch, 3, 1, 1),
                nn.BatchNorm2d(next_ch),
                self.act_fn,
            ]

        arr = arr + [
            nn.Conv2d(channels[-1], 3, 3, 1, 1),
            nn.Tanh()
        ]
        self.some = nn.Sequential(*arr)
        
    def forward(self, x):
        return self.some(x)


print(Model3()(torch.randn(1, 3, 128, 128)).shape)