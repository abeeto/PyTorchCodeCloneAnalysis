from env_GoTogether import EnvGoTogether
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class P_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(P_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_score = self.fc3(x)
        return F.softmax(action_score, dim=-1)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class AC():
    def __init__(self, state_dim, action_dim):
        super(AC, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.p_net = P_net(state_dim, action_dim)
        self.q_net = Q_net(state_dim, action_dim)
        self.gamma = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-2)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=1e-3)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.p_net.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def train_AC(self, state_list, action_list, prob_list, reward_list, next_state_list):
        state = state_list[0]
        next_state = next_state_list[0]
        for i in range(1, len(state_list)):
            state = np.vstack((state, state_list[i]))
            next_state = np.vstack((next_state, next_state_list[i]))
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        next_a_prob = self.p_net.forward(next_state)
        for epoch in range(5):
            q = self.q_net.forward(state)
            next_q = self.q_net.forward(next_state)
            expect_q = q.clone()
            for i in range(len(state_list)):
                expect_q[i, action_list[i]] = reward_list[i] + self.gamma * torch.sum(next_a_prob[i, :] * next_q[i, :])
            loss = self.loss_fn(q, expect_q.detach())
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()

        q = self.q_net.forward(state)
        a_prob = self.p_net.forward(state)
        v = torch.sum(a_prob*q, 1)
        loss = torch.FloatTensor([0.0])
        for i in range(len(state_list)):
            gae = q[i, action_list[i]] - v[i]
            loss -= gae * torch.log(torch.FloatTensor([prob_list[i]]))
        loss = loss/len(state_list)
        self.p_optimizer.zero_grad()
        loss.backward()
        self.p_optimizer.step()

if __name__ == '__main__':
    state_dim = 4
    action_dim = 4
    max_epi = 1000
    max_mc = 500
    epi_iter = 0
    mc_iter = 0
    acc_reward = 0
    reward_curve = []
    env = EnvGoTogether(15)
    agent = AC(state_dim, action_dim)
    for epi_iter in range(max_epi):
        state_list = []
        action_list = []
        prob_list = []
        reward_list = []
        next_state_list = []
        for mc_iter in range(max_mc):
            state = env.get_state()
            action, action_prob = agent.get_action(state)
            group_list = [action, 2]
            reward, done = env.step(group_list)
            next_state = env.get_state()
            acc_reward += reward
            state_list.append(state)
            action_list.append(action)
            prob_list.append(action_prob)
            reward_list.append(reward)
            next_state_list.append(next_state)
            if done:
                break
        agent.train_AC(state_list, action_list, prob_list, reward_list, next_state_list)
        print('epi', epi_iter, 'reward', acc_reward / mc_iter, 'MC', mc_iter)
        env.reset()
        acc_reward = 0