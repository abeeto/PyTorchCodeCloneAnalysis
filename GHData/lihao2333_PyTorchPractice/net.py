import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self._conv_1 = nn.Conv2d(3, 6, 5)
        self._pool = nn.MaxPool2d(2,2)
        self._conv_2 = nn.Conv2d(6, 16, 5)

        self._fc_1 = nn.Linear(16 * 5 * 5, 120)
        self._fc_2 = nn.Linear(120, 84)
        self._fc_3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self._pool(F.relu(self._conv_1(x)))
        x = self._pool(F.relu(self._conv_2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self._fc_1(x))
        x = F.relu(self._fc_2(x))
        x = self._fc_3(x)
        return x
