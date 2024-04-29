import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import time
from collections import deque
import random
import gym
from torchviz import make_dot, make_dot_from_trace

#Hyperparamters

lr = 0.0005
max_steps = 500 #maximum for cartpole_v1
episodes = 200
discount = 0.95
hidden_size = 32
batch_size = 32
mse = False
print_outputs = False
num_experiments = 10

#DQN-specific parameters

capacity = 2000
epsilon_start = 0.99
epsilon_decay = 1000
epsilon_min = 0.1
epsilon = epsilon_start

#Initializing the gym environment

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#Defining the neural network

def initializer(param):
  torch.nn.init.xavier_uniform(param)

class Net(nn.Module):

    def __init__(self):

        #Pytorch has bias layers built in. In tensorflow, we might add a bias layer separately.
        
        super(Net, self).__init__()

        #Instantiates and initializes the hidden layer
        self.hidden_layer = nn.Linear(state_size,hidden_size)
        initializer(self.hidden_layer.weight)

        #Instantiates and initializes the layer that computes state-action values
        self.action_layer = nn.Linear(hidden_size,action_size)
        initializer(self.action_layer.weight)

    def forward(self,x):

        #X is a state for which the network computes state-action values
        x = F.relu(self.hidden_layer(x))
        x = self.action_layer(x)
        return x

#Defining the memory module

class Memory():

  def __init__(self):
    self.data = []

  def push(self,new):

    self.data.append(new)

    if len(self.data) > capacity:
      amount = len(self.data) - capacity
      self.data[0:amount] = []
    
  def __len__(self):
    return len(self.data)

  def sample(self):
    new_data = random.sample(self.data,batch_size)
    return new_data

#Preprocessing method

def preprocess(states):
  return torch.FloatTensor(states)

#Training loop

def train():

    sample = memory.sample()
    states,actions,rewards,nextstates,done = zip(*sample)


    states = torch.stack(states)
    nextstates = torch.stack(nextstates)
    actions = torch.stack(actions).view(batch_size,1)
    rewards = torch.tensor(rewards)

    q_values = net(states).gather(1,actions.long())#torch.sum(net(states) * torch.FloatTensor(one_hot_actions,device=device).unsqueeze(1).detach(),dim=2)

    with torch.no_grad():
        target = rewards + discount*(net(nextstates).max(1)[0].detach()) + (-1 * torch.FloatTensor(done) - 1)

    optimizer.zero_grad()
    if mse:
      loss = criterion(q_values,target.unsqueeze(1))
    else:
      loss = F.smooth_l1_loss(q_values,target.unsqueeze(1))
    loss.backward()
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

experiment_scores = []

for e in range(num_experiments):

  experiment_scores = []

  #Execution and training loop
  net = Net()
  memory = Memory()
  total_rewards = []

  #Defining the optimizer and loss function

  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), lr=lr)

  for i in range(episodes):

    state = preprocess(env.reset())
    done = False
    episode_reward = 0 #running reward for the episode

    for t in range(max_steps):

      with torch.no_grad():
        if epsilon < np.random.rand():
          action = net(state).argmax(0)
        else:
          action = torch.tensor(np.random.randint(0,action_size))
          epsilon_rate = (epsilon_start - epsilon_min) / epsilon_decay
          epsilon = max(epsilon_min, epsilon - epsilon_rate)

      nextState, reward, done, _ = env.step(action.item())
      nextState = preprocess(nextState)
      episode_reward += reward
      memory.push((state, action, reward, nextState, done))
      state = nextState
      
      if done:
        if print_outputs:
          print('Episode ' + str(i) + ' finished with reward ' + str(episode_reward) + ' with epsilon ' + str(epsilon))
        total_rewards.append(episode_reward)
        break

      if len(memory) >= batch_size:

        train()
  
  experiment_scores += total_rewards

#Graph results

print("Average loss: " + str(sum(experiment_scores)/len(experiment_scores)))
print("Final training episode graph:")
plt.plot(total_rewards)
plt.show()

