import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, action_size, seed=3):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7*4, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values"""

        x = self.bn1(F.relu(self.conv1(state)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.view(-1, 64*7*7*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x