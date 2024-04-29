import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 2)  # Prob of Left

    def forward(self, x):
        # x = x.unsqueeze(0)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        # print(self.fc3(x).size()) # 2
        x = F.softmax(self.fc3(x), dim=0)
        # print(x.size())
        return x


def main():

    # Plot duration curve:
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Parameters
    num_episode = 5000
    batch_size = 10
    learning_rate = 0.01
    gamma = 0.99

    env = gym.make('CartPole-v0')
    policy_net = PolicyNet()
    # optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Batch History
    reward_pool = []
    logprob_pool = []
    steps = 0


    for e in range(num_episode):

        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        # env.render(mode='rgb_array')

        # Interaction
        for t in count():

            probs = policy_net(state)
            # print(probs.size())
            m = Categorical(probs)
            action = m.sample()
            logprob_pool.append(m.log_prob(action))

            action = action.data.numpy().astype(int)
            next_state, reward, done, _ = env.step(action)
            # env.render(mode='rgb_array')

            # To mark boundarys between episodes
            if done:
                reward = 0

            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()

            steps += 1

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        # Update policy

        if e > 0 and e % batch_size == 0:
            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            mean_loss = 0
            for i in range(steps):
                reward = reward_pool[i]
                loss = -logprob_pool[i] * reward  # Negtive score function x reward
                mean_loss = mean_loss + loss

            mean_loss = mean_loss / steps
            mean_loss.backward()

            optimizer.step()

            logprob_pool = []
            reward_pool = []
            steps = 0


if __name__ == '__main__':
    main()
