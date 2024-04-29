import torch as T
import torch.nn as nn
import gym
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

env = gym.make('CartPole-v0')

class PolicyGradientNetwork(nn.Module):
    def __init__(self,lr,fc1_dims,fc2_dims,n_actions,input_dims):
        super(PolicyGradientNetwork,self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)

        try:
            self.device = T.device('cuda:0')
        except:
            self.device = T.device('cpu:0')

        self.to(self.device)

    def forward(self,obs):
        state = T.Tensor(obs).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self,input_dims,fc1_dims=32,fc2_dims=32,lr=0.001,gamma=0.95,n_actions=2):
        self.gamma = gamma
        self.policy = PolicyGradientNetwork(lr,fc1_dims,fc2_dims,n_actions,input_dims)
        self.action_memory = []
        self.reward_memory = []

    def choose_action(self,state):
        probs = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probs)   ## creates a probability chart for getting the value selected chance
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory,dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for i in range(t,len(self.reward_memory)):
                G_sum += self.reward_memory[i] * discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        G = T.tensor(G,dtype=T.float).to(self.policy.device)
        loss = 0

        for g,logp in zip(G,self.action_memory):
            loss += -g * logp
        
        loss.backward()
        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []

