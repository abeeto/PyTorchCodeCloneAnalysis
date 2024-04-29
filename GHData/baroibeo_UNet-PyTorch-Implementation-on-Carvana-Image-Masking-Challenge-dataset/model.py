import torch
import torch.nn as nn

class ConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,conv_configs=[64,128,256,512]):
        super(UNet,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.down_convs=self.create_downs(conv_configs)
        self.up_conv_tranpose=self.create_ups(conv_configs)[0]
        self.up_conv_blocks=self.create_ups(conv_configs)[1]
        self.bottleneck=ConvBlocks(conv_configs[-1],conv_configs[-1]*2)
        self.last_conv=nn.Conv2d(conv_configs[0],out_channels,kernel_size=1,stride=1,padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)

    
    def forward(self,x):

        #list for saving down layers to concatenate later
        downsaving=[]
        
        #Down process
        for down in self.down_convs:
            x=down(x)
            downsaving.append(x)
            x=self.maxpool(x)
        
        #Bottleneck
        x=self.bottleneck(x)

        #Up
        downsaving=downsaving[::-1] #Reverse downsaving list to concat from bottom to head
        for i,up in enumerate(self.up_conv_tranpose):
            x=up(x) #ConvTranpose2d
            x=torch.cat([downsaving[i],x],dim=1) #Concat
            x=self.up_conv_blocks[i](x) #Conv2d
            # print(x.shape)

        x=self.last_conv(x)
        return x

    def create_downs(self,conv_configs):
        layers=nn.ModuleList()
        in_channels=self.in_channels
        for config in conv_configs:
            layers.append(ConvBlocks(in_channels,config))
            in_channels=config
        return layers

    def create_ups(self,conv_configs):
        convtranpose_layers=nn.ModuleList()
        convblocks_layers=nn.ModuleList()
        for config in conv_configs[::-1]:
            convtranpose_layers.append(nn.ConvTranspose2d(config*2,config,kernel_size=2,stride=2))
            convblocks_layers.append(ConvBlocks(config*2,config))
        return convtranpose_layers,convblocks_layers

if __name__=='__main__':
    x=torch.randn(size=(3,3,256,256))
    unet=UNet(in_channels=3,out_channels=1)
    out=unet(x)
    print(out.shape)