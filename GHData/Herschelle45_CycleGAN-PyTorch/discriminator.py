import torch 
import torch.nn as nn 

class convblock(nn.Module):
    def __init__(self, inchannels, outchannels, **kwargs):
        super().__init__()
        self.convseq = nn.Sequential(
                    nn.Conv2d(inchannels, outchannels, **kwargs), 
                    nn.InstanceNorm2d(outchannels),
                    nn.LeakyReLU(0.2)
                )
    def forward(self, x):
        return self.convseq(x)

class Discriminator(nn.Module):
    def __init__(self, dfeatures=[3, 64, 128, 256, 512]):
        super().__init__()
        self.convseq = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'), 
                    nn.LeakyReLU(0.2)
                )
        self.disc = nn.Sequential(
                    convblock(dfeatures[1], dfeatures[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect'), 
                    convblock(dfeatures[2], dfeatures[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect' ), 
                    convblock(dfeatures[3], dfeatures[4], kernel_size=4, stride=1, padding=1, padding_mode='reflect'), 
                    convblock(dfeatures[4], 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'), 
                    nn.Sigmoid()
                    )
    def forward(self, x):
        return self.disc(self.convseq(x))
disc = Discriminator()
print(disc)
