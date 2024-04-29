import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# with reference too : https://amaarora.github.io/2020/09/13/unet.html#introduction

'''
UNET architecture consists of 2 components
1. Encoder (Convolutional Operations)
2. Decoder (Transposed Convolutions or Upsampling)
'''


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        '''
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# for debugging
# conv = ConvBlock(3, 64)
# print(conv(torch.randn(1, 3, 572, 572)).shape)


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        '''
        returns output of all conv_layers in conv_blocks 
        which will be used later in decoder for concatenation with ConvTranspose layer 
        '''
        encoder_ops = []
        for block in self.conv_blocks:
            x = block(x)
            encoder_ops.append(x)
            x = self.pool(x)
        return encoder_ops


# encoder_ops = Encoder()
# print(encoder_ops(torch.randn(1, 3, 572, 572))[-1].shape)
# should output torch.Size([1, 1024, 28, 28])


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        # for simplicity using ConvTranspose2d
        self.conv_trans_blocks = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2)
             for i in range(len(chs)-1)]
        )  # e.g. converts [1x1024x30x30] -> [1x512x60x60]
        # original paper uses Upsampling technique which is different than the ConvTranspose2d

        self.conv_blocks = nn.ModuleList(
            [ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )  # conv layers which come after upsampling or ConvTransponse
        # e.g. converts [1x512x60x60] -> [1x1024x60x60]
        self.chs = chs

    # x : [1024x28x28], encoder_ops = [512x64x64, ...,64x568x568] (not including 1024x28x28 since after upsampling we need 512x64x64 to concatenate with skipped layer of that shape)
    def forward(self, x, encoder_ops):
        for i in range(len(self.chs)-1):
            x = self.conv_trans_blocks[i](x)  # ConvTranspose2d

            # cropping to retain the size while concatenating
            encoder_ops[i] = self.crop(encoder_ops[i], x)
            x = torch.cat([x, encoder_ops[i]], dim=1)

            # Convolution Operation after concatenating with skipped layer
            x = self.conv_blocks[i](x)
        return x

    def crop(self, encoder_ops, x):
        '''
        To make sure that before concatenting skipped layers the dimension of 
        input shape of ConvTranspose and output shape of skipped layer must be same
        '''

        _, _, H, W = x.shape
        encoder_ops = torchvision.transforms.CenterCrop([H, W])(encoder_ops)
        return encoder_ops


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64),  num_class=1, retain_dim=False, out_size=(572, 572)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.out_size = out_size
        self.tail = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim  # incase user wants to retain some output size

    def forward(self, x):
        encoder_ops = self.encoder(x)
        # outputs [ 64x568x568, 128x280x280, ..., 1024x28x28 ]
        # but our decoder it accepts in reverse order
        encoder_ops = encoder_ops[::-1]
        # print(encoder_ops[0].shape) [1024x28x28]
        # reason for [1:] is in class Decoder
        output = self.decoder(encoder_ops[0], encoder_ops[1:])
        # so now we have a output of shape [64x392x392]
        # print(decoder_output.shape)

        # final convolution operation
        output = self.tail(output)

        if self.retain_dim:
            output = F.interpolate(output, self.out_size)

        return output


# model = UNet(retain_dim=True, out_size=(256, 256))
# noise = torch.randn(1, 3, 256, 256)
# print(model(noise).shape)
