import torch.nn as nn
import torch.nn.functional as F

''' Mannually picked values (aka tunable parameters)
    kernel_size:  square filter size [kxk]. kernel <-> filter
    stride:       slide of the filter after each operation. [default (1,1) 1 to the right, 1 down]
    out_channels: number of filters. 1 filter => 1 output channel
    out_features: size of the output tensor

    Increase output channels as we add more convolutional layers and shrinkage down the output features
    as we move down the linear layers.

    Data depended parameters:
        in_channels: * of the first conv layer == number of colour channels in train images
                     * of the other conv layers == with the output of the previous one
        in_features: * of the output layer == number of classes in train set
                     * of the other linear layers == with the output of the previous one

    When we change from conv layers to a linear layer we flattern our output

    For conv layers, filters ARE the learnable parameters (aka weights) rank-4 tesnor [out_channels, in_channels, kernel_size, kernel_size]
    For linear layers, we have weight matrices rank-2 tensor with shape [out_features, in_features]
        * they map an (in_features)-D space to an (out_features)-D space using a weight matrix
'''


class Network(nn.Module):  # build Neural Networks we extend torch.nn.Module
    def __init__(self):
        """ Class constructor """
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # linear layers or Fully Connected or dense
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        """ Function to implement the forward pass of the tensors for our (custom) Network """
        # (1) input layer : f(x) = x
        t = t                   # torch.Size( [1, 1, 28, 28] )

        # (2) hidden conv layer
        t = self.conv1(t)       # torch.Size( [1, 6, 24, 24] )
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # torch.Size( [1, 6, 12, 12] )
        # (3) hidden conv layer
        t = self.conv2(t)       # torch.Size( [1, 12, 8, 8] )
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # torch.Size( [1, 12, 4, 4] )

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)  # torch.Size( [1, 192] )
        t = self.fc1(t)            # torch.Size( [1, 120] )
        t = F.relu(t)
        # (5) hidden linear layer
        t = self.fc2(t)           # torch.Size( [1, 60] )
        t = F.relu(t)

        # (6) output linear layer
        t = self.out(t)          # torch.Size( [1, 10 ] )
        # t = F.softmax(t, dim=1) , cause of the cross entropy loss function being used (already compute softmax)

        return t
