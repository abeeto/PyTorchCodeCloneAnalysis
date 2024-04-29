import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.convolution = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7,stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear = nn.Linear(in_features=5408,out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        convolution_out = self.convolution.forward(x)
        relu_out = self.relu(convolution_out)
        maxpool_out = self.maxpool.forward(relu_out)
        maxpool_out_flattened = torch.flatten(maxpool_out,start_dim=-3, end_dim=-1)
        outs = self.linear(maxpool_out_flattened)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
