import torch 
import torch.nn as nn 

class cblock(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size, stride, padding, act=True, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(inchannels, outchannels, kernel_size, stride, padding, padding_mode='reflect') if downsample 
                    else nn.ConvTranspose2d(inchannels, outchannels, kernel_size, stride, padding, output_padding=1), 
                    nn.InstanceNorm2d(outchannels), 
                    nn.ReLU() if act else nn.Identity()
                )
    def forward(self, x):
        return self.conv(x)

class resb(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resblock = nn.Sequential(
                    cblock(channels, channels, 3, 1, 1),
                    cblock(channels, channels, 3, 1, 1, False) 
                )
    def forward(self, x):
        return x+self.resblock(x)

class Generator(nn.Module):
    def __init__(self, channels, gfeatures):
        super().__init__()
        self.initial = cblock(channels, gfeatures, 7, 1, 3)
        self.down = nn.Sequential(
                    cblock(gfeatures, gfeatures*2, 3, 2, 1),
                    cblock(gfeatures*2, gfeatures*4, 3, 2, 1),
                )
        self.resblocks = nn.Sequential(*[resb(gfeatures*4) for _ in range(9)])
        self.up = nn.Sequential(
                    cblock(gfeatures*4, gfeatures*2, kernel_size=3, stride=2 , padding=1, act=True, downsample=False), 
                    cblock(gfeatures*2, gfeatures, kernel_size=3, stride=2, padding=1, act=True, downsample=False), 
                    nn.Conv2d(gfeatures, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
                )
    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(self.down(x))
        return self.up(x) 
gen = Generator(3, 64)
print(gen)
