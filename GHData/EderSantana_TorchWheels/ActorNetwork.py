import numpy as np
import math
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        super(ActorNetwork, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.h0 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        self.brake = nn.Linear(HIDDEN2_UNITS, 1)

        self.steering.weight.data.normal_(0., 1e-4)
        self.acceleration.weight.data.normal_(0., 1e-4)
        self.brake.weight.data.normal_(0., 1e-4)

    def forward(self, x):
        x = F.relu(self.h0(x))
        x = F.relu(self.h1(x))
        s = F.tanh(self.steering(x))
        a = F.sigmoid(self.acceleration(x))
        b = F.sigmoid(self.brake(x))
        return torch.cat((s, a, b), 1)


def train_actor_target(model, target):
    # target.h0.weight = nn.Parameter(torch.from_numpy(
    #     model.h0.cpu().weight.data.numpy() * model.TAU + (1 - model.TAU) * 
    #     target.h0.cpu().weight.data.numpy()))
    # target.h1.weight = nn.Parameter(torch.from_numpy(
    #     model.h1.cpu().weight.data.numpy() * model.TAU + (1 - model.TAU) * 
    #     target.h1.cpu().weight.data.numpy()))
    # target.cuda()
    target.h0.weight.data = model.h0.weight.data * model.TAU + (1-model.TAU)*target.h0.weight.data
    target.h1.weight.data = model.h1.weight.data * model.TAU + (1-model.TAU)*target.h1.weight.data
    return target
