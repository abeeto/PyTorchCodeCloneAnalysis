from torch import nn
import torch.nn.functional as F

__build__ = 2018
__author__ = "singsam_jam@126.com"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 320)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
