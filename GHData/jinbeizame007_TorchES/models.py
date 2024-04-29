import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(3, 20)
        self.l2 = nn.Linear(20, 1)
    
    def forward(self, x):
        h = torch.tanh(self.l1(x))
        h = self.l2(h)
        return h
    
    def get_action(self, x):
        x = torch.FloatTensor(x).view(1, -1)
        action = self.forward(x).clamp(min=-1, max=1).numpy()[0]
        return action