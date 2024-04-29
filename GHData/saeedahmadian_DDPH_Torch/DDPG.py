import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from GridTest import Grid
import pandapower.networks as pn
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def moving_average(signal,N=20):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(signal, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return moving_ave


class ReplayBuffer:
    def __init__(self, capacity, limit=200, Ns=70, Na=9):
        self.capacity = capacity
        self.buffer = []
        # np.empty([limit, 2*Ns+Na+2])
        self.position = 0
        self.Ns= Ns
        self.Na = Na
        self.limit = limit

    def push(self, state, action, reward, next_state, done):
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # self.buffer[self.position] = (state, action, reward, next_state, done)
        # self.buffer[self.position]= np.concatenate([state, action, reward, next_state, done],axis=0)
        self.buffer.append(np.concatenate([state, action, reward, next_state, done], axis=0))
        # self.position = (self.position + 1) % self.capacity
        self.position = self.position + 1
        if self.position == self.limit:
            self.position = 0
            self.buffer=self.buffer[self.limit-41:]
    def sample(self, batch_size):
        half_batch = round(batch_size/2)
        data= np.array(self.buffer)
        ind = data[:,self.Ns+self.Na].argsort()
        data = data[ind[::-1]]
        batch=data[0:batch_size,:]
        batch2 = np.array(random.sample(list(data[half_batch:batch_size+1,:]), half_batch))
        # batch= np.concatenate([batch1,batch2],axis=0)

        state = batch[:, 0:self.Ns]
        action = batch[:, self.Ns:self.Ns+self.Na]
        reward = batch[:, self.Ns+self.Na:self.Ns+self.Na+1]
        next_state = batch[:, self.Ns+self.Na+1:self.Ns+self.Na+1+self.Ns]
        done = batch[:,self.Ns+self.Na+1+self.Ns]
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions:
    def __init__(self,upp=1,low=-1):
        self.upp = upp
        self.low = low

    def _action(self, action):
        low_bound = self.low
        upper_bound = self.upp

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action):
        low_bound = self.low
        upper_bound = self.upp

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.low = -1
        self.high = 1
        self.reset()

    def reset(self):
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


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size1=128,hidden_size2=256,hidden_size3=512,hidden_size4=32, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear_act = nn.Linear(num_actions, hidden_size1)
        self.linear_var = nn.Linear(num_inputs , hidden_size1)
        # self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.normalized = nn.BatchNorm1d(hidden_size1,affine=False)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size3)
        self.linear3 = nn.Linear(hidden_size3, hidden_size4)
        self.linear4 = nn.Linear(hidden_size4, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # x = torch.cat([state, action], 1)

        x = F.relu(self.linear_act(action)+self.linear_var(state))
        x = self.normalized(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]


def ddpg_update(batch_size,value_net,policy_net,target_value_net,target_policy_net,
                policy_optimizer,value_optimizer,
                replay_buffer,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)
    # ss = torch.reshape(state, [-1, 70])
    value = value_net(state, action)

    criterion=nn.MSELoss()
    value_loss = criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return value_net,policy_net,target_value_net,target_policy_net,policy_optimizer,value_optimizer,value_loss,policy_loss



env = Grid(pn.case5())
Ns=env.StateFeatures()[0]*env.StateFeatures()[1]
state_dim = Ns
action_dim = env.ActionFeature()
ou_noise = OUNoise(action_dim)

hidden_dim = 256

value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)


best_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
best_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
path = '/model'

target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_lr = 1e-3
policy_lr = 1e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)



replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 24000
max_steps = 20
frame_idx = 0
rewards = []
batch_size = 40
reward_com=[]

best_reward_per_ep = 0

value_net_loss=[]
policy_net_loss=[]

if len(os.getcwd()+'/model_new') !=0:
    value_net = torch.load(os.getcwd()+'/model_new/best_value_net')
    policy_net = torch.load(os.getcwd()+'/model/best_policy_net')
    target_value_net = torch.load(os.getcwd()+'/model/best_value_net')
    target_policy_net = torch.load(os.getcwd()+'/model/best_policy_net')
    best_value_net = torch.load(os.getcwd()+'/model/best_value_net')
    best_policy_net = torch.load(os.getcwd()+'/model/best_policy_net')
    value_net.eval()
    policy_net.eval()
    target_value_net.eval()
    target_policy_net.eval()
    best_value_net.eval()
    best_policy_net.eval()


while frame_idx < max_frames:
    env.reset()
    # state = env.InitState().reshape([-1, Ns])
    state = env.InitState().reshape(Ns)
    ind = np.random.choice(np.arange(env.net.line.shape[0]))
    env.Attack(ind)
    ou_noise.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        next_state, reward, done,reward_comps = env.take_action(action)
        reward_com.append(reward_comps)
        next_state = next_state.reshape(Ns)
        reward = np.array(reward).reshape(1)
        done = np.array(done).reshape(1)
        # next_state= next_state.reshape([-1,Ns])

        # action= np.array(action).reshape([-1,action_dim])
        replay_buffer.push(state, action, reward, next_state, done)

        if replay_buffer.position % batch_size == 0:
            value_net, policy_net, target_value_net, target_policy_net,policy_optimizer, value_optimizer,v_loss,p_loss= \
                ddpg_update(batch_size,value_net,policy_net,target_value_net,target_policy_net,
                            policy_optimizer,value_optimizer,replay_buffer)
            value_net_loss.append(float(v_loss.item()))
            policy_net_loss.append(float(p_loss.item()))
            # ddpg_update(batch_size, value_net, policy_net, target_value_net, target_policy_net,
            #             policy_optimizer, value_optimizer, replay_buffer)
        state = next_state
        episode_reward += reward

        if done[0]==1:
            if episode_reward > best_reward_per_ep:
                best_reward_per_ep = episode_reward
                for best_param, param in zip(best_value_net.parameters(), value_net.parameters()):
                    best_param.data.copy_(param.data)

                for best_param, param in zip(best_policy_net.parameters(), policy_net.parameters()):
                    best_param.data.copy_(param.data)

                torch.save(best_value_net, os.getcwd()+'/model_new/best_value_net')
                torch.save(best_policy_net, os.getcwd()+'/model_new/best_policy_net')

            print('In Episode {}, step {}, we reached to terminal with total episode reward function : {} '
                  'and best reward is : {}'.
                  format(frame_idx, step,episode_reward,best_reward_per_ep))
            break

    frame_idx += 1
    if frame_idx % 100 == 0:
        np.save('reward', rewards)
        np.save('reward_com', reward_com)
        print('In Episode: {}, the reward function is {}'.format(frame_idx, episode_reward))

    rewards.append(episode_reward)


a= 1