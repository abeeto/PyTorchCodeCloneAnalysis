import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):

    def __init__(
        self,
        conv_dim = 32
        ):
        super().__init__()

        # Store parameters.
        self.conv_dim = conv_dim

        # Conv2D layers
        self.conv1 = nn.Conv2d(3, self.conv_dim, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.conv_dim, self.conv_dim*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.conv_dim*2, self.conv_dim*4, 4, 2, 1, bias=False)

        # Leaky ReLU with a negative slope of 0.1
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Batch Normalization
        self.batch2 = nn.BatchNorm2d(self.conv_dim*2)
        self.batch3 = nn.BatchNorm2d(self.conv_dim*4)

        # Fully-connected layer for the last layer
        self.fc_out = nn.Linear(4*4*self.conv_dim*4, 1)

    def forward(
        self, 
        x
        ):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.leaky_relu(x)        
        x = x.view(-1, 4*4*self.conv_dim*4)

        x = self.fc_out(x)

        return x

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):


        return