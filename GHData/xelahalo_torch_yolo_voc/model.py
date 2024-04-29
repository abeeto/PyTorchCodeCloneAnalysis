import torch
import torch.nn as nn

""" 
The architecture based on the paper: You Only Look Once: Unified, Real-Time Object Detection
by Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

Tuple is structured by (kernel_size, filters, stride, padding) 

"M" is maxpooling with stride 2x2 and kernel 2x2

List is [tuple, tuple, n] n being the number of repeats

See architecture.png
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
"""
Class representing a block in the Convolutional Neural Network

Leaky ReLU was used in the paper as the activation function.

(BatchNorm was not used in the paper. It is used to normalize the output of the previous layers.
 Using batch normalization learning becomes efficient also it can be used as regularization to 
 avoid overfitting of the model).
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyRelu(self.batchNorm(self.conv(x)))

"""
Yolo (V1) the class representing the architecture itself.

It is used to fully connect the layers into the neural network.
"""

class Yolo(nn.Module): 
    def __init__(self, in_channels=3, **kwargs):
        super(Yolo, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fcs(torch.flatten(x, start_dim=1))


    """
    Using the configuration to build the layers
    """
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]

                # We set the next in_channel to the filter size
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    # same as above
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    """
    Creating the fully connected layers
    """
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), #In the original papers this was 4096, but it would've taken a lot of time
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )