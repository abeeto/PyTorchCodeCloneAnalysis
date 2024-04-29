import torch
import torch.nn as nn
import torch.nn.functional as F 

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = 1)

    def forward(self, x):
        res = x
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        out = res + x2
        return out

class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 2):
        super(DownConv, self).__init__()
        self.down_conv = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = 1)

    def forward(self, x):
        out = F.elu(self.down_conv(x))
        return out      

class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 1, stride = 1):
        super(UpConv, self).__init__()
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride)

    def forward(self, x):
        x1 = self.up_sample(x)
        out = self.conv(x1)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, in_channel = 21, last_layer_channel = 128, hidden_channel = 128, resblock_num = 6, min_feature_map_size = 8, z_num = 64):
        super(UnetGenerator, self).__init__()
        self.resblock_num = resblock_num
        self.hidden_channel = hidden_channel
        self.min_feature_map_size = min_feature_map_size
        self.encoder_layer_list = []

        #conv after input
        self.in_conv = nn.Conv2d(in_channel, hidden_channel, kernel_size = 3, stride = 1, padding = 1)
        #conv before output
        self.out_conv = nn.Conv2d(2 * last_layer_channel, 3, kernel_size = 3, stride = 1, padding = 1)
        #fc to get latent space z
        self.fc_encode = nn.Linear(resblock_num * hidden_channel * min_feature_map_size * min_feature_map_size, z_num)
        self.fc_decode = nn.Linear(z_num, resblock_num * hidden_channel * min_feature_map_size * min_feature_map_size)

        self.resblock_encode1 = ResBlock(hidden_channel * 1, hidden_channel * 1)
        self.resblock_encode2 = ResBlock(hidden_channel * 2, hidden_channel * 2)
        self.resblock_encode3 = ResBlock(hidden_channel * 3, hidden_channel * 3)
        self.resblock_encode4 = ResBlock(hidden_channel * 4, hidden_channel * 4)
        self.resblock_encode5 = ResBlock(hidden_channel * 5, hidden_channel * 5)
        self.resblock_encode6 = ResBlock(hidden_channel * 6, hidden_channel * 6)

        self.down1 = DownConv(hidden_channel * 1, hidden_channel * 2)
        self.down2 = DownConv(hidden_channel * 2, hidden_channel * 3)
        self.down3 = DownConv(hidden_channel * 3, hidden_channel * 4)
        self.down4 = DownConv(hidden_channel * 4, hidden_channel * 5)
        self.down5 = DownConv(hidden_channel * 5, hidden_channel * 6)

        self.resblock_decode1 = ResBlock(2 * hidden_channel * (resblock_num-0), 2 * hidden_channel * (resblock_num-0))
        self.resblock_decode2 = ResBlock(2 * hidden_channel * (resblock_num-1), 2 * hidden_channel * (resblock_num-1))
        self.resblock_decode3 = ResBlock(2 * hidden_channel * (resblock_num-2), 2 * hidden_channel * (resblock_num-2))
        self.resblock_decode4 = ResBlock(2 * hidden_channel * (resblock_num-3), 2 * hidden_channel * (resblock_num-3))
        self.resblock_decode5 = ResBlock(2 * hidden_channel * (resblock_num-4), 2 * hidden_channel * (resblock_num-4))
        self.resblock_decode6 = ResBlock(2 * hidden_channel * (resblock_num-5), 2 * hidden_channel * (resblock_num-5))

        self.up1 = UpConv(2 * hidden_channel * (resblock_num-0), hidden_channel * (resblock_num-1))
        self.up2 = UpConv(2 * hidden_channel * (resblock_num-1), hidden_channel * (resblock_num-2))
        self.up3 = UpConv(2 * hidden_channel * (resblock_num-2), hidden_channel * (resblock_num-3))
        self.up4 = UpConv(2 * hidden_channel * (resblock_num-3), hidden_channel * (resblock_num-4))
        self.up5 = UpConv(2 * hidden_channel * (resblock_num-4), hidden_channel * (resblock_num-5))      


    def forward(self, x):
        print ('in_conv')
        x = self.in_conv(x)
        print(x.size())

        x = self.resblock_encode1(x)
        self.encoder_layer_list.append(x)
        x = self.down1(x)
        print (x.size())

        x = self.resblock_encode2(x)
        self.encoder_layer_list.append(x)
        x = self.down2(x)
        print (x.size())

        x = self.resblock_encode3(x)
        self.encoder_layer_list.append(x)
        x = self.down3(x)
        print (x.size())

        x = self.resblock_encode4(x)
        self.encoder_layer_list.append(x)
        x = self.down4(x)
        print (x.size())

        x = self.resblock_encode5(x)
        self.encoder_layer_list.append(x)
        x = self.down5(x)
        print (x.size())

        x = self.resblock_encode6(x)
        self.encoder_layer_list.append(x)
        print (x.size())

        x = torch.reshape(x, [-1, self.resblock_num*self.hidden_channel*
                              self.min_feature_map_size*self.min_feature_map_size])          

        print ('fc_encode')
        print (x.size())
        z = self.fc_encode(x)
        print (z.size())

        print ('fc_encode')
        x = self.fc_encode(z)

        x = torch.reshape(x, [-1, self.resblock_num*self.hidden_channel, 
                              self.min_feature_map_size, self.min_feature_map_size])
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 1]), 1)
        x = self.resblock_decode1(x)
        x = self.up1(x)
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 2]), 1)
        x = self.resblock_decode2(x)
        x = self.up2(x)
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 3]), 1)
        x = self.resblock_decode3(x)
        x = self.up3(x)
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 4]), 1)
        x = self.resblock_decode4(x)
        x = self.up4(x)
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 5]), 1)
        x = self.resblock_decode5(x)
        x = self.up5(x)
        print (x.size())

        x = torch.cat((x, self.encoder_layer_list[self.resblock_num - 6]), 1)
        x = self.resblock_decode6(x)
        print (x.size())

        print ('out_conv')
        out = self.out_conv(x)

        return out





