"""
Assignment 2, COMP338 - Step 1. Define a Convolutional Neural Network

Thepnathi Chindalaksanaloet, 201123978
Robert Szafarczyk, 201307211
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        batch_layer = nn.BatchNorm2d(64)
        self.batch_norm_and_relu = lambda x : F.relu(batch_layer(x))

        self.fc = nn.Linear(57600, 5)


    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 250, 250)
        the comments ignore the batch size (it stays the same accross all layers)
        """
        # When applying a kernel, the shape changes: (N - F + 2P)/S + 1

        # First hidden layer: a convolution layer with a filter size 7x7, stride 2, padding 3,
        # the number of channels 64, followed by Batch Normalization and ReLu.
        # shape : 3x250x250 -> 64x125x125 = (250 - 7 + 2*6)/2 + 1 = 128
        x = self.conv1(x)
        x = self.batch_norm_and_relu(x)

        # Second hidden layer: max pooling with a filter size 3x3, stride 2, padding 0;
        # 64x125x125 -> 64x62x62
        x = self.pooling_layer(x)

        # Third hidden layer: a convolution layer with a filter size 3x3, stride 1, padding 1,
        # the number of channels 64, followed by Batch Normalization and ReLu.
        # 64x62x62 -> 64x62x62
        x = self.conv2(x)
        x = self.batch_norm_and_relu(x)

        # Fourth hidden layer: max pooling with a filter size 3x3, stride 2, padding 0;
        # 64x62x62 -> 64x30x30
        x = self.pooling_layer(x)

        # Fully connected layer, with the output channel 5 (i.e., the number of classes);
        # 64x30x30 -> 57600 (Reshape data for the fully connected layer).
        x = x.view(-1, 64 * 30 * 30)
        # 57600 -> 5
        x = self.fc(x)

        # Return raw, unnormalised scores for each class. nn.CrossEntropyLoss() will apply the
        # softmax function to normalise the scores in the range [0,1].
        return x


def createLossAndOptimizer(net, learning_rate=0.001):
    # it combines softmax with negative log likelihood loss
    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return criterion, optimizer
