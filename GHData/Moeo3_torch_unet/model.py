from torch.nn import Sequential, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid, ModuleList
import torch

class Conv33(Module):
    def __init__(self, channels_in, channels_out, acti):
        super(Conv33, self).__init__()
        self.conv_33 = Sequential(
            Conv2d(channels_in, channels_out, 3, padding=1, padding_mode='reflect'),
            BatchNorm2d(channels_out),
            acti
        )
        pass
    def forward(self, inputs):
        return self.conv_33(inputs)

class ConvBlock(Module):
    def __init__(self, channels_in, channels_out, acti):
        super(ConvBlock, self).__init__()
        self.conv_block = Sequential(
            Conv33(channels_in, channels_out, acti),
            Conv33(channels_out, channels_out, acti)
        )
    def forward(self, inputs):
        return self.conv_block(inputs)

class DownBlock(Module):
    def __init__(self, channels_in, channels_out, acti):
        super(DownBlock, self).__init__()
        self.down_block = Sequential(
            MaxPool2d(2),
            ConvBlock(channels_in, channels_out, acti)
        )
    def forward(self, inputs):
        return self.down_block(inputs)
    
class UpBlock(Module):
    def __init__(self, channels_in, channels_out, acti):
        super(UpBlock, self).__init__()
        self.up = ConvTranspose2d(channels_in, channels_in // 2, 2, 2)
        self.conv = ConvBlock(channels_in, channels_out, acti)
    def forward(self, inputs, concatenate_part):
        return self.conv(torch.cat([concatenate_part, self.up(inputs)], 1))



class UNet(Module):
    def __init__(self, channels_in = 1, channels_out = 1, conv_start_channels = 64, depth = 4, acti = ReLU(inplace=True)):
        super(UNet, self).__init__()
        self.depth = depth

        self.in_block = ConvBlock(channels_in, channels_in * conv_start_channels, acti)

        self.down_blocks = ModuleList()
        self.up_blocks = ModuleList()

        # self.down_blocks = Sequential()
        # self.up_blocks = Sequential()

        # self.down_list = []
        for i in range(depth):
            down_block = DownBlock(channels_in * conv_start_channels * pow(2, i), channels_in * conv_start_channels * pow(2, i + 1), acti)
            self.down_blocks.add_module(str(i), down_block)
            # self.down_list.append(DownBlock(channels_in * conv_start_channels * pow(2, i), channels_in * conv_start_channels * pow(2, i + 1), acti))

        # self.up_list = []
        for i in range(depth, 0, -1):
            up_block = UpBlock(channels_in * conv_start_channels * pow(2, i), channels_in * conv_start_channels * pow(2, i - 1), acti)
            self.up_blocks.add_module(str(depth - i), up_block)
            # self.up_list.append(UpBlock(channels_in * conv_start_channels * pow(2, i), channels_in * conv_start_channels * pow(2, i - 1), acti))
        
        self.out_block = Sequential(
            Conv2d(conv_start_channels, channels_out, 1),
            Sigmoid()
        )
          
    def forward(self, inputs):
        next_inputs = self.in_block(inputs)
        concatenate_list = [next_inputs]
        for i in range(self.depth):
            next_inputs = self.down_blocks[i](next_inputs)
            concatenate_list.append(next_inputs)
        for i in range(self.depth):
            next_inputs = self.up_blocks[i](next_inputs, concatenate_list[self.depth - i - 1])
        return self.out_block(next_inputs)