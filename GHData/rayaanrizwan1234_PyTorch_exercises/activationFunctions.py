import torch
import torch.nn as nn
import torch.nn.functional as F

class NeauralNet():
    def __init__(self, inputSize, hiddenSize):
        super(NeauralNet, self).__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, 1)

    def forward(self, X):
        out = F.relu(self.linear1(X))
        out = F.sigmoid(self.linear2(out))
        return out