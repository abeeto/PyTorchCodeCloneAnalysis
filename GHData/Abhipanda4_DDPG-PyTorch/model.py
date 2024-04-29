import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Actor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit):
        super(Actor, self).__init__()
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    # Critic estimates the state-value function Q(S, A)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400 + action_dim, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, 1)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        x = torch.cat((s, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
