from numpy import pad
import torch
import torch.nn as nn

# generator model based on UNet architecture 

class unetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False):
        super().__init__()
        self.convLayer = nn.Sequential(
             # convolutional or transposed convolutional layers (down- or up-sampling is done directly by the layer using stride > 1)
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)
            )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.convLayer(x)
        return self.dropout(x) if self.use_dropout == True else x



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
       
