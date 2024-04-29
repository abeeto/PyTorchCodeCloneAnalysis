import torch
import torch.nn as nn

# PyTorch implementation by vinesmsuic
# The paper claimed to use BatchNorm and Leaky ReLu.
# But here we use InstanceNorm instead of BatchNorm.

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, num_features=64, num_residuals=8):
        super().__init__()
        self.initial = nn.Sequential(
            #k7n64s1
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="zeros"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        #Down-convolution
        self.down_blocks = nn.ModuleList(
            [
                #k3n128s2
                nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode="zeros"),
                #k3n128s1
                nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*2),
                nn.ReLU(inplace=True),

                #k3n256s2
                nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1, padding_mode="zeros"),
                #k3n256s1
                nn.Conv2d(num_features*4, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*4),
                nn.ReLU(inplace=True),
            ]
        )

        #8 residual blocks => 8 times [k3n256s1, k3n256s1]
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up_blocks = nn.ModuleList(
            [
                #k3n128s1/2
                nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode="zeros"),
                #k3n128s1
                nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*2),
                nn.ReLU(inplace=True),

                #k3n64s1/2
                nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode="zeros"),
                #k3n64s1
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        )

        #Convert to RGB
        #k7n3s1
        self.last = nn.Conv2d(num_features*1, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="zeros")

    def forward(self, x):
        x = self.initial(x)
        #Down-convolution
        for layer in self.down_blocks:
            x = layer(x)
        #8 residual blocks
        x = self.res_blocks(x)
        #Up-convolution
        for layer in self.up_blocks:
            x = layer(x)
        #Convert to RGB
        x = self.last(x)
        #TanH
        return torch.tanh(x)
        
def test():
    in_channels = 3
    img_size = 256
    x = torch.randn((2, in_channels, img_size, img_size))
    gen = Generator(in_channels)
    print(gen(x).shape)

if __name__ == "__main__":
    test()


