# discriminator model based on PatchGAN architecture 

from turtle import st
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    # con
    def __init__(self, n_inputs, n_outputs, kernel_size=4, stride=2):
        super().__init__()  # calls constructor for nn.Module superclass

        # conv layer
        self.convLayer = nn.Sequential(
            nn.Conv2d(n_inputs, n_outputs, kernel_size, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(n_outputs),
            nn.LeakyReLU(0.2),
        )

    # propagate an input through the layer
    def forward(self, x):
            return self.convLayer(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_maps_sizes=[64, 128, 256, 512]):
        super().__init__()

        # initial layer slightly different from basic block, it takes two stacked RGB images as input (since it's a CGAN, both x and y)
        self.initLayer = nn.Sequential(
            nn.Conv2d(in_channels*2, feature_maps_sizes[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        # stack the rest of the network
        layers=[]
        in_channels = feature_maps_sizes[0]
        for fmap_size in feature_maps_sizes[1:]:
            layers.append(
                ConvBlock(in_channels, fmap_size, stride=1 if fmap_size == feature_maps_sizes[-1] else 2),
            )
            in_channels = fmap_size   # inputs of the current are outputs of the previous
        
        # last layer for single channel patchGAN output (inputs 512 fmaps and outputs single channel)
        layers.append(
            nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        # build sequential model from layers list
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        # concatenate input and output images along first dimension (which is the channel number)
        x = torch.cat([x, y], dim=1)
        # propagate
        x = self.initLayer(x)
        return self.model(x)



# dimensional testing with random input tensors
def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    # get prediction grid
    res = model.forward(x, y)
    print(res.size())

if __name__== "__main__":
    test()
