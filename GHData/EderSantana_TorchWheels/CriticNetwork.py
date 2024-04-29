import numpy as np
import math
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        super(CriticNetwork, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.h0 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h2 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.h4 = nn.Linear(HIDDEN2_UNITS, action_size)

    def forward(self, x, a):
        w = F.relu(self.h0(x))
        w = F.relu(self.h1(w))

        a1 = F.relu(self.h2(a))

        h = w + a1
        h = F.relu(self.h3(h))

        v = self.h4(h)

        return v 


def train_critic_target(model, target):
    for a, b in zip([model.h0, model.h1, model.h2, model.h3, model.h4],
                 [target.h0, target.h1, target.h2, target.h3, target.h4]):
        #b.weight.data = torch.from_numpy(
        #    a.cpu().weight.data.numpy() * model.TAU + (1 - model.TAU) *
        #    b.cpu().weight.data.numpy())
        b.weight.data = a.weight.data * model.TAU + (1 - model.TAU) * b.weight.data
    return target
