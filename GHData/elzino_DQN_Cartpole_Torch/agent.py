from collections import namedtuple
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CnnQLearning(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_output_size(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        def function_call_repeat(f, time, *args):
            if time == 1:
                return f(*args)
            else:
                return f(function_call_repeat(f, time - 1, *args))
        convw = function_call_repeat(conv2d_output_size, 3, w)
        convh = function_call_repeat(conv2d_output_size, 3, h)
        linear_output_size = convh * convw * 32
        self.head = nn.Linear(linear_output_size, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DnnQLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 10)
        self.linear2 = nn.Linear(10, 5)
        self.head = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu((self.linear1(x)))
        x = F.relu(self.linear2(x))
        return self.head(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(policy_net, state, steps_done, eps_start=0.9, eps_end=0.05, eps_decay=0.005):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done * eps_decay)
    if sample >= eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], dtype=torch.long).to(device)
