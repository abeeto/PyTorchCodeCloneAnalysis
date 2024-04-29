from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3 * 112 * 112, 128)
        self.bmi_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bmi_out(x)
        x = F.relu(x) + 15.0

        return x

