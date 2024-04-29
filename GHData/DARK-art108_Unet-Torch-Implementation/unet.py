import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(inplace=True)
    )

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down_conv_1 = double_conv(1, 64)
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, 
                                             out_channels=512, 
                                             kernel_size=2, 
                                             stride=2
                                            )

        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, 
                                             out_channels=256, 
                                             kernel_size=2, 
                                             stride=2
                                            )      
        
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, 
                                             out_channels=128, 
                                             kernel_size=2, 
                                             stride=2
                                            )

        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, 
                                             out_channels=64, 
                                             kernel_size=2, 
                                             stride=2
                                            )

        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, image):
        x1 = self.down_conv_1(image) # 568
        x2 = self.max_pool_1(x1) # 284
        x3 = self.down_conv_2(x2) # 280
        x4 = self.max_pool_1(x3) # 140
        x5 = self.down_conv_3(x4) # 136
        x6 = self.max_pool_1(x5) # 68
        x7 = self.down_conv_4(x6) # 64
        x8 = self.max_pool_1(x7) # 32
        x9 = self.down_conv_5(x8) # 28

        # upsampling
        x = self.up_trans_1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))
        x = self.up_trans_2(x)
        y = crop_tensor(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))
        x = self.up_trans_3(x)
        y = crop_tensor(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))
        x = self.up_trans_4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv_4(torch.cat([x, y], dim=1))

        out = self.out(x)

        return out

if __name__ == '__main__':
    image = torch.randn(1, 1, 572, 572)
    unet = Unet()
    print(unet(image))
