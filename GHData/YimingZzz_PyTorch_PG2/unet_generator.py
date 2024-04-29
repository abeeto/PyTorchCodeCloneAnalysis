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
        #conv after input
        self.in_conv = nn.Conv2d(in_channel, hidden_channel, kernel_size = 3, stride = 1, padding = 1)
        #conv before output
        self.out_conv = nn.Conv2d(2 * last_layer_channel, 3, kernel_size = 3, stride = 1, padding = 1)
        #fc to get latent space z
        self.fc_encode = nn.Linear(resblock_num * hidden_channel * min_feature_map_size * min_feature_map_size, z_num)
        self.fc_decode = nn.Linear(z_num, resblock_num * hidden_channel * min_feature_map_size * min_feature_map_size)
                
        self.down = []
        self.resblock_encode = []
        self.resblock_decode = []
        self.up = []
        self.encoder_layer_list = []
        for i in range (self.resblock_num):
            if i < self.resblock_num - 1:
                self.down.append(DownConv(hidden_channel * (i+1), hidden_channel * (i+2)))
                self.up.append(UpConv(2 * hidden_channel * (resblock_num-i), hidden_channel * (resblock_num-i-1)))
            self.resblock_encode.append(ResBlock(hidden_channel * (i+1), hidden_channel * (i+1)))
            self.resblock_decode.append(ResBlock(2 * hidden_channel * (resblock_num-i), 2 * hidden_channel * (resblock_num-i)))

    def forward(self, x):
        #encode
        print ('in_conv')
        x = self.in_conv(x)
        print (x.size())
        
        for i in range(self.resblock_num):
            print (i)
            print (self.resblock_encode[i])
            x = self.resblock_encode[i](x)
            print (x.size())
            self.encoder_layer_list.append(x)
            if i < (self.resblock_num - 1):
                x = self.down[i](x)
            print (x.size())
            

        x = torch.reshape(x, [-1, self.resblock_num*self.hidden_channel*
                              self.min_feature_map_size*self.min_feature_map_size])    
        print ('fc_encode')
        print (self.fc_encode)
        print (x.size())
        z = self.fc_encode(x)
        print (z.size())

        #decode
        print ('fc_decode')
        x = self.fc_decode(z)
        x = torch.reshape(x, [-1, self.resblock_num*self.hidden_channel, 
                              self.min_feature_map_size, self.min_feature_map_size])
        print (x.size())
        
        for i in range(self.resblock_num):
            print (i)
            print (self.resblock_decode[i])
    
            x = torch.cat((x, self.encoder_layer_list[self.resblock_num-i-1]), 1)
            print(x.size())
            x = self.resblock_decode[i](x)
            if i < (self.resblock_num - 1):
                x = self.up[i](x)
            print (x.size())
            
        print ('out_conv')
        out = self.out_conv(x)

        return out