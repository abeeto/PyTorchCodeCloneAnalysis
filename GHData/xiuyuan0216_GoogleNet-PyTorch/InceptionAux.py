from BasicConv import *
from Inception import *

import torch
import torch.nn as nn
import torch.nn.functional as F 

class InceptionAux(nn.Module):
    
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super().__init__()
        self.conv = BasicConv(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, output_size=4)
        out = self.conv(out)
        out = out.view(-1, 128*4*4)
        out = self.fc1(out)
        out = nn.ReLU()
        out = self.dropout(out)
        out = self.fc2(out)
        return out
