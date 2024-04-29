from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class DQNNET(nn.Module):
    def __init__(self, state_shape, action_shape, lr=1e-3):
        super(DQNNET, self).__init__()
        self.fc1 = nn.Linear(in_features=state_shape, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=action_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
