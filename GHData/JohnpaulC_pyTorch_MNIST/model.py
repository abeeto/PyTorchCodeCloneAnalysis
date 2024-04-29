import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        # input channel, output channel, Filter Size, No Change size
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        # Filter Size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.dropout = nn.Dropout(p = 0.4)

        # input nodes, output nodes
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, apply_softmax = False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Change to 1 dimensional Tensor
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if apply_softmax:
            y_pred = F.softmax(x, dim=1)
        else:
            y_pred = x

        return y_pred