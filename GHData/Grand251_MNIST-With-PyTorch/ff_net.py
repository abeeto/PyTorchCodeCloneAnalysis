import torch.nn.functional as F
import torch.nn as nn


class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
        return y_pred
