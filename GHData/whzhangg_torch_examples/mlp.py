import torch
from torch import nn
import torch.nn.functional as F

# module
class MultiLinear(nn.Module):
    """ linear layer network with given depth and hidden units
    input (Cin,) -> (Cout,)"""
    def __init__(self, 
        input_length: int, 
        output_length: int,
        linearlayer_connection: tuple,
        activation: nn.Module = None
    ):
        super().__init__()

        self.output_length = output_length
        self.input_length = input_length
        prev_connection = self.input_length

        self.activation = activation or nn.ReLU()

        linear_sequence = []
        for nconnection in linearlayer_connection:
            linear_sequence.append(nn.Linear(prev_connection, nconnection))
            linear_sequence.append(self.activation)
            prev_connection = nconnection
        linear_sequence.append(nn.Linear(prev_connection, output_length))  
        
        self.linearlayer = nn.Sequential(*linear_sequence)

    @property
    def input_shape(self):
        return self.input_length

    @property
    def output_shape(self):
        return self.output_length

    def forward(self, x):
        out = self.linearlayer(x)
        return out


class ConvNet(nn.Module):
    """hyper parameters:
        number of convolution layer [1, 2, 3, 4]
        dropout1: (0 - 0.7) 
        dropout2: 
        channel_increase [8, 16, 32, 64]
        linear unit: [16, 32, 64, 128, 256]
    """
    input_pixel   = 28
    input_channel = 1
    output_size   = 10

    def __init__(self, 
        nconvolution: int = 2,
        initial_channel: int = 32,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
        n_linear: int = 128
    ):
        super().__init__()

        convs = []
        input_channel = self.input_channel
        for i in range(nconvolution):
            if i == 0:
                output_channel = initial_channel
            else:
                output_channel = input_channel * 2
            convs.append(nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, 1, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout1)
            ))
            input_channel = output_channel

        
        self.convolutions = torch.nn.ModuleList(convs)
        # image is output_channel * 28 * 28
        self.pooling = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(dropout2)
        self.fc1 = nn.Linear(int(output_channel * self.input_pixel ** 2 / 4), n_linear)
        self.fc2 = nn.Linear(n_linear, self.output_size)

        self.c1 = nn.Conv2d(1, 32, 3, 1, padding= 1)
        self.c2 = nn.ReLU()
        self.c3 = nn.Dropout(dropout1)

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
