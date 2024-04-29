import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + noises * stds
    actions = torch.tanh(us)

    log_pis = calculate_log_pi(log_stds), noises, actions

    return actions, log_pis


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        x = self.fc3(x)
        x = x.chunk(2, dim=-1)[0]
        return x

    def sample(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        means, log_stds = x.chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.fc4 = nn.Linear(state_dim+action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        
        x1 = F.relu(self.fc1(x), inplace=True)
        x1 = F.relu(self.fc2(x1), inplace=True)
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x), inplace=True)
        x2 = F.relu(self.fc5(x2), inplace=True)
        x2 = self.fc6(x2)

        return x1, x2


