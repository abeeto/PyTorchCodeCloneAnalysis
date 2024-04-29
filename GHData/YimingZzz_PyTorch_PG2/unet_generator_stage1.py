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
        x2 = F.elu(self.conv2(x1))
        out = res + x2
        return out

class EncodeDown(nn.Module):
    def __init__(self, res_in_channel, res_out_channel, down_out_channel, kernel_size = 3, stride = 2):
        super(EncodeDown, self).__init__()
        self.resblock_encode = ResBlock(res_in_channel, res_out_channel)
        self.down_conv = nn.Conv2d(res_out_channel, down_out_channel, kernel_size = kernel_size, stride = stride, padding = 1)

    def forward(self, x):
        x_encode = self.resblock_encode(x)
        out = F.elu(self.down_conv(x_encode))
        return out, x_encode      
    
class EncodeNoDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncodeNoDown, self).__init__()
        self.encode = ResBlock(in_channel, out_channel)
    def forward(self, x):
        out = self.encode(x)
        return out
        
class DecodeUp(nn.Module):
    def __init__(self, res_in_channel, res_out_channel, up_out_channel, kernel_size = 1, stride = 1):
        super(DecodeUp, self).__init__()
        self.resblock_decode = ResBlock(res_in_channel, res_out_channel)
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Conv2d(res_out_channel, up_out_channel, kernel_size = kernel_size, stride = stride)

    def forward(self, x1, x2):      
        x = torch.cat([x1, x2], dim = 1)
        x_decode = self.resblock_decode(x)
        x_up= self.up_sample(x_decode)
        out = F.elu(self.conv(x_up))
        return out

class DecodeNoDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecodeNoDown, self).__init__()
        self.decode = ResBlock(in_channel, out_channel)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        out = self.decode(x)
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
        
        self.encode_down1 = EncodeDown(hidden_channel * 1, hidden_channel * 1, hidden_channel * 2)
        self.encode_down2 = EncodeDown(hidden_channel * 2, hidden_channel * 2, hidden_channel * 3)
        self.encode_down3 = EncodeDown(hidden_channel * 3, hidden_channel * 3, hidden_channel * 4)
        self.encode_down4 = EncodeDown(hidden_channel * 4, hidden_channel * 4, hidden_channel * 5)
        self.encode_down5 = EncodeDown(hidden_channel * 5, hidden_channel * 5, hidden_channel * 6)
        self.encode = EncodeNoDown(hidden_channel * 6, hidden_channel * 6)
        
        self.decode_up1 = DecodeUp(2 * hidden_channel * (resblock_num - 0), 2 * hidden_channel * (resblock_num - 0), hidden_channel * (resblock_num -1))
        self.decode_up2 = DecodeUp(2 * hidden_channel * (resblock_num - 1), 2 * hidden_channel * (resblock_num - 1), hidden_channel * (resblock_num -2)) 
        self.decode_up3 = DecodeUp(2 * hidden_channel * (resblock_num - 2), 2 * hidden_channel * (resblock_num - 2), hidden_channel * (resblock_num -3))    
        self.decode_up4 = DecodeUp(2 * hidden_channel * (resblock_num - 3), 2 * hidden_channel * (resblock_num - 3), hidden_channel * (resblock_num -4))
        self.decode_up5 = DecodeUp(2 * hidden_channel * (resblock_num - 4), 2 * hidden_channel * (resblock_num - 4), hidden_channel * (resblock_num -5))
        self.decode = DecodeNoDown(2 * hidden_channel * (resblock_num - 5), 2 * hidden_channel * (resblock_num - 5))

    def forward(self, condition, x):
        #print ('in_conv')
        x_input = torch.cat([condition, x], dim = 1)

        x = F.elu(self.in_conv(x_input))
        #print(x.size())

        x, x_encode1 = self.encode_down1(x)
        #print (x.size())

        x, x_encode2 = self.encode_down2(x)
        #print (x.size())

        x, x_encode3 = self.encode_down3(x)
        #print (x.size())

        x, x_encode4 = self.encode_down4(x)
        #print (x.size())

        x, x_encode5 = self.encode_down5(x)
        #print (x.size())

        x_encode6 = self.encode(x)
        #print (x.size())

        x = torch.reshape(x_encode6, [-1, self.resblock_num*self.hidden_channel*
                              self.min_feature_map_size*self.min_feature_map_size])          

        #print ('fc_encode')
        #print (x.size())
        z = self.fc_encode(x)
        #print (z.size())

        #print ('fc_decode')
        x = self.fc_decode(z)

        x_reconstruct = torch.reshape(x, [-1, self.resblock_num*self.hidden_channel, 
                              self.min_feature_map_size, self.min_feature_map_size])
        #print (x_reconstruct.size())

        x = self.decode_up1(x_reconstruct, x_encode6)
        #print (x.size())
        
        x = self.decode_up2(x, x_encode5)
        #print (x.size())

        x = self.decode_up3(x, x_encode4)
        #print (x.size())

        x = self.decode_up4(x, x_encode3)
        #print (x.size())

        x = self.decode_up5(x, x_encode2)
        #print (x.size())
        
        x = self.decode(x, x_encode1)
        #print (x.size())

        #print ('out_conv')
        out = F.tanh(self.out_conv(x))
        

        return out