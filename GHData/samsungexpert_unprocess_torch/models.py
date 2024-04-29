import torch
import torch.nn as nn

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

class DemosaicNet(nn.Module):
    def __init__(self): 
        super().__init__() 

        self.depth_to_space = nn.functional.pixel_shuffle
        self.space_to_depth = space_to_depth
        self.relu = nn.ReLU()

        self.model = nn.Sequential(
            self.__conv_block(4,12, 3),
            self.__conv_block(12,12,3),
            self.__conv_block(12,12, 3),
            self.__conv_block(12,12,3),
            self.__conv_block(12,12,3),
        )

    
    def forward(self, x): 
        
        x_std = self.space_to_depth(x, 2)       ## (B, C, H, W ) --> (B, 4C , H//2, W//2 )
        x_1 = self.relu(self.model[0](x_std))  
        x_2 = self.relu(self.model[1](x_1))
        x_3 = self.relu(self.model[2](x_2))
        x_4 = self.relu(self.model[3](x_3))
        x_5 = self.relu(self.model[4](x_4))
        
        x_6 = x_5 + x_1 

        y = self.depth_to_space(x_6, 2)        ## (B, 4C, H//2, W//2 ) --> (B, C, H, W)

        return y

        
        

    def __conv_block(self, in_features, out_features, kernel_size, upsample=False):
        if upsample:
            # 8
            conv = nn.ConvTranspose2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=1)
        else:
            conv = nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size-1)//2)

        # 9
        return conv

    
if __name__ == "__main__":
    x = torch.rand((1, 1, 256, 256))
    demosaic_net = DemosaicNet()

    print("G(x) shape:", demosaic_net(x).shape)
