import torch
import torch.nn as nn


class LeNetModel(nn.Module):
    """
    Defining the LeNet Architecture consisting of two Convolutional Layers followed by MaxPooling Layers further
    connected to two fully connected layers. The activation fucntion used is ReLU activation function, and the model
    returns the logsoftmax probabilities as the output.
    """

    def __init__(self, input_channels, output_classes):
        super(LeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 20, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(800, 300)
        self.fc2 = nn.Linear(300, output_classes)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.relu(self.fc1(torch.flatten((x, 1))))
        x = self.logsoftmax(self.fc2(x))
        return x