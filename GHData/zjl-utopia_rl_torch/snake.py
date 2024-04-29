import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time
from collections import deque
import numpy as np

from agents.DQNAgent import DQNAgent


env = gym.make('gym_custom:Snake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.state.shape[0] * env.state.shape[1]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # confirm the shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, 16)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(16, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # Define how the input data pass inside the network
        # x = x.flatten()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 80x80x3
        self.features = nn.Sequential(
            # 40x40x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # 20x20x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # 10x10x256
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
agent = DQNAgent(n_state=N_STATES, n_action=N_ACTIONS, action_shape=ENV_A_SHAPE, network=CNNNet,
                 use_target_net=True)

# Start training
ep_rs = deque(maxlen=3)
print("\nCollecting experience...")
for i_episode in range(10000000):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        # take action based on the current state
        # s = s.flatten()
        s = np.transpose(np.resize(s, (80, 80, 3)), (2, 0, 1))
        a = agent.select_action(s, deterministic=i_episode > 1000)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)
        # s_ = s_.flatten()
        s_ = np.transpose(np.resize(s_, (80, 80, 3)), (2, 0, 1))

        # store the transitions of states
        agent.store_transition(s, a, r, s_, done)

        ep_r += r
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if agent.ready_to_learn():
            agent.learn()
            if done:
                ep_rs.append(ep_r)
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

        if done:
            # if game is over, then skip the while loop.
            break
        # use next state to update the current state.
        s = s_

        if len(ep_rs) == 3 and np.mean(ep_rs) >= 1:
            time.sleep(0.2)
        # time.sleep(0.3)
