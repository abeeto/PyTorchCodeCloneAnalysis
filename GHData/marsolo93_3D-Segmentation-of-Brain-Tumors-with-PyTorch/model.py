import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride, padding, bn=True, act=True, conv=True, rev=False):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.bn = bn
        self.act = act
        self.conv = conv
        self.rev = rev
        self.conv_ = nn.Conv3d(self.in_filters, self.out_filters, kernel_size, stride, padding=padding)
        if self.bn:
            if self.rev:
                self.bn_ = nn.GroupNorm(self.in_filters // 2, self.in_filters)
            else:
                self.bn_ = nn.GroupNorm(self.out_filters//2, self.out_filters)
        if self.act:
            self.act_ = nn.LeakyReLU()

    def forward(self, x):
        if self.rev:
            if self.bn:
                x = self.bn_(x)
            if self.act:
                x = self.act_(x)
            if self.conv:
                x = self.conv_(x)
        else:
            if self.conv:
                x = self.conv_(x)
            if self.bn:
                x = self.bn_(x)
            if self.act:
                x = self.act_(x)
        return x

class ConvTransposeBlock3D(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride, padding):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.conv_ = nn.ConvTranspose3d(self.in_filters, self.out_filters, kernel_size, stride, padding=padding)

    def forward(self, x):
        x = self.conv_(x)
        return x

class ConvLevel3D(nn.Module):

    def __init__(self, in_filters, out_filters, last=False):
        super().__init__()
        self.conv_0 = ConvBlock3D(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bn=False, act=False)
        self.conv_1 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_2 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_3 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_4 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True, conv=False)
        self.conv_halfing = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout3d(p=0.5)
        self.last = last

    def forward(self, x):
        x = self.conv_0(x)
        short = x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.drop(x)
        #x = self.conv_3(x)
        skip = x + short
        skip = nn.LeakyReLU()(skip)
        x = self.conv_4(skip)
        if self.last:
            return x
        else:

            x = self.conv_halfing(x)
            return x, skip

class ConvUpsampleLevel3D(nn.Module):

    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv_0 = ConvBlock3D(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
        self.conv_transpose = ConvTransposeBlock3D(out_filters, out_filters, kernel_size=2, stride=2, padding=0)
        self.conv_1 = ConvBlock3D(out_filters * 2, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_2 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_3 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True)
        self.conv_4 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1, rev=True, conv=False)
        self.conv_5 = ConvBlock3D(out_filters, out_filters, kernel_size=3, stride=1, padding=1)

        self.drop = nn.Dropout3d(p=0.6)

    def forward(self, x, skip):
        x_short = x
        x = self.conv_0(x)
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        short = x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.drop(x)
        #x = self.conv_3(x)
        x = self.conv_4(x + short)
        x = self.conv_5(x)
        return x, x_short


class Encoder(nn.Module):

    def __init__(self, in_filter, out_filter):
        super().__init__()
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.encoder_length = len(out_filter)
        self.conv_lvl_list = nn.ModuleList()

        for i in range(self.encoder_length):
            if i == 0:
                self.conv_lvl_list.append(ConvLevel3D(self.in_filter, self.out_filter[i]))
            elif i == self.encoder_length - 1:
                self.conv_lvl_list.append(ConvLevel3D(self.out_filter[i - 1], self.out_filter[i], last=True))
            else:
                self.conv_lvl_list.append(ConvLevel3D(self.out_filter[i - 1], self.out_filter[i]))


    def forward(self, x):
        self.skip_list = []
        for i in range(len(self.conv_lvl_list)):
            if i == self.encoder_length - 1:
                skip = self.conv_lvl_list[i](x)
            else:
                x, skip = self.conv_lvl_list[i](x)
            self.skip_list.append(skip)
        return self.skip_list

# class SkipMixer(nn.Module):
#     def __init__(self, filter_list):
#         super().__init__()
#         self.constant = filter_list[0] * 128



class Decoder(nn.Module):

    def __init__(self, filter_list, num_classes):
        super().__init__()
        self.filter_list = filter_list[::-1]
        self.num_classes = num_classes
        self.decoder_length = len(self.filter_list)

        self.conv_transpose_lvl_list = nn.ModuleList()
        self.conv_short_list = nn.ModuleList()
        for i in range(self.decoder_length - 1):
            self.conv_transpose_lvl_list.append(ConvUpsampleLevel3D(self.filter_list[i], self.filter_list[i+1]))

            self.conv_short_list.append(ConvBlock3D(self.filter_list[i], self.num_classes, kernel_size=1, stride=1, padding=0, bn=False, act=False))

    def forward(self, skip_list):
        skip_list = skip_list[::-1]
        x = skip_list[0]
        for i in range(self.decoder_length - 1):
            x, short = self.conv_transpose_lvl_list[i](x, skip_list[i+1])
            if i == 0:
                short_added = self.conv_short_list[i](short)
                short_added = nn.Upsample(scale_factor=2)(short_added)
            else:
                short = self.conv_short_list[i](short)
                short_added = short_added + short
                short_added = nn.Upsample(scale_factor=2)(short_added)
        return x, short_added

class SemanticHead(nn.Module):

    def __init__(self, in_filters, num_classes):
        super().__init__()
        self.conv_1 = ConvBlock3D(in_filters, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, short):
        return self.conv_1(x) + short

class UNet3D(nn.Module):
    def __init__(self, in_channels, filter_list, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.filter_list = filter_list
        self.num_classes = num_classes

        self.encoder = Encoder(self.in_channels, self.filter_list)
        self.decoder = Decoder(self.filter_list, self.num_classes)
        self.head = SemanticHead(self.filter_list[0], self.num_classes)

    def forward(self, x):
        skip_list = self.encoder(x)
        decoder_out, short = self.decoder(skip_list)
        out = self.head(decoder_out, short)
        return out

if __name__ == '__main__':
    import numpy as np
    test = torch.tensor(np.random.normal(0, 1, [1, 1, 128, 128, 128]))
    test = test.double()
    num_classes = 4
    filter_list = [16, 32, 64, 128, 256]
    model = UNet3D(1, filter_list, num_classes)
    model = model.to('cuda')
    model = model.float()
    out = model(test.to('cuda').float())
    print(out.shape)
















