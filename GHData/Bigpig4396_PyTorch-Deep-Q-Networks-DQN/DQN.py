from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_SingleCatchPigs import EnvSingleCatchPigs

class ReplayMemory(object):
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.is_av = False
        self.batch_size = 64

    def remember(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def is_available(self):
        if len(self.memory) > self.batch_size:
            self.is_av = True
        return self.is_av

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()

        self.n_action = n_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.flat1 = Flatten()
        self.fc1 = nn.Linear(288, 20)
        self.fc2 = nn.Linear(20, self.n_action)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.flat1(h)
        # print(h.size())
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h

class Agent(object):
    def __init__(self, n_action):
        self.n_action = n_action
        self.dqn = DQN(self.n_action)
        self.gamma = 0.95
        self.loss_fn = torch.nn.MSELoss()

    def get_action(self, obs, epsilon):
        if random.random() > epsilon:
            q_value = self.dqn.forward(self.img_to_tensor(obs).unsqueeze(0))
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.randint(0, self.n_action - 1)
        return action

    def img_to_tensor(self, img):
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def list_to_batch(self, x):
        # transform a list of image to a batch of tensor [batch size, input channel, width, height]

        temp_batch = self.img_to_tensor(x[0])
        temp_batch = temp_batch.unsqueeze(0)
        for i in range(1, len(x)):
            img = self.img_to_tensor(x[i])
            img = img.unsqueeze(0)
            temp_batch = torch.cat([temp_batch, img], dim=0)
        return temp_batch

    def train(self, x):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        next_obs_list = []
        for i in range(len(x)):
            obs_list.append(x[i][0])
            action_list.append(x[i][1])
            reward_list.append(x[i][2])
            done_list.append(x[i][3])
            next_obs_list.append(x[i][4])

        obs_list = self.list_to_batch(obs_list)
        next_obs_list = self.list_to_batch(next_obs_list)

        q_list = self.dqn.forward(obs_list)
        next_q_list = self.dqn.forward(next_obs_list)

        next_q_list_max_v, next_q_list_max_i = next_q_list.max(1)
        expected_q_value = q_list.clone()
        for i in range(len(x)):
            temp_index = next_q_list_max_i[i].item()
            expected_q_value[i][action_list[i]] = reward_list[i] + self.gamma * next_q_list[i][temp_index]

        loss = self.loss_fn(q_list, expected_q_value.detach())

        self.dqn.optimizer.zero_grad()
        loss.backward()
        self.dqn.optimizer.step()


if __name__ == '__main__':
    env = EnvSingleCatchPigs(7)
    max_MC_num = 30000
    dqn = Agent(5)
    memory = ReplayMemory()
    for MC_iter in range(max_MC_num):
        print('iter', MC_iter)
        obs = env.get_obs()
        action = dqn.get_action(obs, 1.0-(MC_iter/max_MC_num))
        reward, bool_done = env.step(action)
        if bool_done:
            done = 1.0
            print('episode finishes')
            env.reset()
        else:
            done = 0.0
        next_obs = env.get_obs()
        memory.remember(obs, action, reward, done, next_obs)
        if memory.is_available():
            dqn.train(memory.sample())

    env.reset()
    for MC_iter in range(max_MC_num):
        env.plot_scene()
        obs = env.get_obs()
        action = dqn.get_action(obs, 0)
        reward, bool_done = env.step(action)