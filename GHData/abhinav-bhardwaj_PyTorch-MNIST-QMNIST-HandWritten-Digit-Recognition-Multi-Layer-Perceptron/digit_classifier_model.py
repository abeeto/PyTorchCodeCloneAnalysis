from torch.nn import Module
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F

class SudokuNet(Module):
  def __init__ (self):
    super().__init__()
    self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5,5))
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

    self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

    self.fc1 = Linear(in_features=800, out_features=500)
    self.relu3 = ReLU()

    self.fc2 = Linear(in_features=500, out_features=10)
    self.logSoftmax = LogSoftmax(dim=1)


  def forward(self, X):
    X = self.conv1(X)
    X = self.relu1(X)
    X = self.maxpool1(X)

    X = self.conv2(X)
    X = self.relu2(X)
    X = self.maxpool2(X)

    X = flatten(X,1)
    X = self.fc1(X)
    X = self.relu3(X)
    X = self.fc2(X)
    res = self.logSoftmax(X)
    return res
