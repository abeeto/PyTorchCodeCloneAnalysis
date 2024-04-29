
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self,in_channel, out_channel,k=3, s=1,isGenerator=True):
        super(ConvolutionalBlock, self).__init__()
        self.isGenerator = isGenerator
        if self.isGenerator:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,k,s,1),
                nn.BatchNorm2d(out_channel),
                nn.PReLU(out_channel),
                nn.Conv2d(in_channel, out_channel,k,s,1),
                nn.BatchNorm2d(out_channel),
            )
        if not self.isGenerator:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,k,s),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        if self.isGenerator:
            return x + self.block(x)

        return self.block(x)

class GeneratorNetwork(nn.Module):
    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        self.inputBlock = nn.Sequential(
            nn.Conv2d(3,64,9,1,4),
            nn.SiLU()
        )#dim = [N,64,16,16]
        resblocks = [ConvolutionalBlock(64,64,3,1) for __ in range(5)]
        self.residualBlocks = nn.Sequential(
            *resblocks,
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64)
        )
        self.pixelShufflers = nn.Sequential(
            nn.Conv2d(64,256,3,1,1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,256,3,1,1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(),
            nn.Conv2d(64,3,9,1,4)
        )
    def forward(self, x):
        inp = self.inputBlock(x)
        x = self.residualBlocks(inp) + inp
        return self.pixelShufflers(x)


class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3,64,3,1),
            nn.LeakyReLU(0.2)
        )
        discBlocks = []
        in_channel, out_channel = 64,64
        for block in range(1,8):
            stride = block%2 +1
            discBlocks.append(
                ConvolutionalBlock(in_channel,out_channel,3,s=stride, isGenerator=False)
            )
            in_channel = out_channel
            if block%2==1:
                out_channel = out_channel*2
        self.discBlocks = nn.Sequential(*discBlocks)
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(6*6*512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            #Sigmoid can ve dismissed. We will use BCEwithLogitsLoss for adversarial loss

        )
    def forward(self, x):
        x = self.input(x)
        x = self.discBlocks(x)
        return self.classify(x)

