import torch
import torch.nn.functional as F
from torch import nn


class MNISTClassifier(nn.Module):
    """
    Reference : https://github.com/pytorch/serve/blob/master/examples/image_classifier/mnist/mnist.py
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) # (B, 1, 28, 28) => (B, 32, 26, 26)
        x = F.relu(x)
        x = self.conv2(x) # (B, 32, 26, 26) => (B, 64, 24, 24)
        x = self.max_pooling(x) # (B, 64, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # (B, 64*12*12) = (B, 9216)
        x = self.fc1(x) # (B, 9216) => (B, 128)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # (B, 128) => (B, 10)
        output = F.log_softmax(x, dim=-1) # (B, 10) => (B, 10)

        return output