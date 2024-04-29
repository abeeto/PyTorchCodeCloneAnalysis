"""
Definition of the network
"""

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Inherited from the parent class nn.Module
        super(Net, self).__init__()
        # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Non-Linear ReLU Layer: max(0,x)
        self.relu = nn.ReLU()
        # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
