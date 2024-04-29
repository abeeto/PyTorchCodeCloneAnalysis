import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import copy
import numpy as np
from collections import deque
import random
from torch.autograd import Variable


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self) -> object:
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# Ornstein-Ulhenbeck Process
# Taken from https://gist.github.com/cyoon1729/2ea43c5e1b717cc072ebc28006f4c887#file-utils-py
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class CriticNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes=None):
        super(CriticNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ln = []  # LayerNormalization

        if hidden_layer_sizes is None:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layer_sizes[0])])
            self.ln.append(nn.LayerNorm(hidden_layer_sizes[0]).to(self.device))
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
                self.ln.append(nn.LayerNorm(hidden_layer_sizes[i+1]).to(self.device))
            self.layers.append(nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes) - 1], output_size))

    def forward(self, state, action):

        if len(self.layers) == 1:
            x = torch.cat([state, action], 1)
            return self.layers[0](x)

        else:
            x = torch.cat([state, action], 1)
            for i in range(len(self.layers) - 1):
                x = F.relu(self.layers[i](x))
                # x = self.ln[i](x)
            return self.layers[len(self.layers) - 1](x)


class ActorNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes=None):
        super(ActorNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ln = []  # LayerNormalization
        self.bn = []  # BatchNormalization

        if hidden_layer_sizes is None:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layer_sizes[0])])
            self.ln.append(nn.LayerNorm(hidden_layer_sizes[0]).to(self.device))
            self.bn.append(nn.BatchNorm1d(hidden_layer_sizes[0]).to(self.device))
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
                self.ln.append(nn.LayerNorm(hidden_layer_sizes[i+1]).to(self.device))
                self.bn.append(nn.BatchNorm1d(hidden_layer_sizes[i+1]).to(self.device))
            self.layers.append(nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes) - 1], output_size))

    def forward(self, state):

        if len(self.layers) == 1:
            return torch.tanh(self.layers[0](state))

        elif len(state.shape) == 1:
            for i in range(len(self.layers) - 1):
                state = F.relu(self.layers[i](state))
                # state = self.ln[i](state)
            return torch.tanh(self.layers[len(self.layers) - 1](state))
        else:
            for i in range(len(self.layers) - 1):
                state = F.relu(self.layers[i](state))
                # state = self.bn[i](state)
            return torch.tanh(self.layers[len(self.layers) - 1](state))


# Thanks Chris Yoon -> https://gist.github.com/cyoon1729/542edc824bfa3396e9767e3b59183cae#file-ddpg-py
class Ddpg:

    def __init__(self, state_dim, action_dim, actor_hid, critic_hid, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=1e-2, replay_size=16384, batch_size=128):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.memory = Memory(replay_size)
        self.discount = gamma
        self.tau = tau

        # The 4 Networks
        self.Actor = ActorNN(input_size=state_dim, output_size=action_dim, hidden_layer_sizes=actor_hid).to(self.device)
        self.Target_Actor = copy.deepcopy(self.Actor).to(self.device)
        self.Critic = CriticNN(input_size=state_dim + action_dim, output_size=action_dim, hidden_layer_sizes=critic_hid).to(self.device)
        self.Target_Critic = copy.deepcopy(self.Critic).to(self.device)

        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=actor_lr)
        self.Critic_optimizer = optim.Adam(self.Critic.parameters(), lr=critic_lr)
        self.Critic_loss = nn.MSELoss()


    def run(self, state):
        state = Variable(torch.from_numpy(state.copy()).float()).to(self.device)
        action = self.Actor.forward(state)
        return action.cpu().detach().numpy()  # if cuda is not available .cpu() is not necessary

    def train(self):

        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Prepare Actor Update
        self.Actor.to(self.device)
        Actor_actions = self.Actor.forward(states)
        Q = self.Critic.forward(states, Actor_actions)
        Actor_update = - Q.mean()

        # Prepare Critic Update
        Q = self.Critic.forward(states, actions)
        Actor_next_actions = self.Target_Actor.forward(next_states)
        Qnext = self.Target_Critic.forward(next_states, Actor_next_actions)
        Critic_update = self.Critic_loss(Q, Qnext * self.discount + rewards)

        # Update Actor NN
        self.Actor_optimizer.zero_grad()
        Actor_update.backward()
        self.Actor_optimizer.step()

        # Update Critic NN
        self.Critic_optimizer.zero_grad()
        Critic_update.backward()
        self.Critic_optimizer.step()

        # Update Target Networks
        for target_param, param in zip(self.Target_Actor.parameters(), self.Actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.Target_Critic.parameters(), self.Critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))







