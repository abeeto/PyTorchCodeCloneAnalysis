import random
import torch
import torch.nn.functional as F

from replay_memory import ReplayMemory
from model import DQNNET
import threading as th


class DQNAgent:
    def __init__(self, observation_space, action_space, device, epsilon_max,
                 epsilon_min, epsilon_decay, memory_capacity, discount=.99, lr=1e-3):

        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.device = device

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_memory = ReplayMemory(memory_capacity)

        self.online_network = DQNNET(
            observation_space.shape[0], action_space.n, lr=lr).to(device)
        self.target_network = DQNNET(
            observation_space.shape[0], action_space.n, lr=lr).to(device)

        self.target_network.eval()
        self.update_target()
        self.isLearning = False

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = torch.argmax(self.online_network(state))
        return action.item()

    def learn(self, batch_size):
        if self.isLearning:
            return
        else:
            self.isLearning = True
            self._learn(batch_size)

    def _learn(self, batch_size):
        try:
            if len(self.replay_memory) < batch_size:
                self.isLearning = False
                return
            states, actions, next_states, rewards, dones = self.replay_memory.sample(
                batch_size, self.device)

            actions = actions.reshape((-1, 1))
            rewards = rewards.reshape((-1, 1))
            dones = dones.reshape((-1, 1))
            actions = actions.long()
            dones = dones.long()
            predicted_qs = self.online_network(states)
            predicted_qs = predicted_qs.gather(1, actions)

            target_qs = self.target_network(next_states)
            target_qs = torch.max(target_qs, dim=1).values
            target_qs = target_qs.reshape(-1, 1)
            target_qs[dones] = 0.1

            y_js = rewards + self.discount * target_qs
            loss = F.smooth_l1_loss(predicted_qs, y_js)
            self.online_network.optimizer.zero_grad()
            loss.backward()
            self.online_network.optimizer.step()
            self.isLearning = False
        except Exception as e:
            print()
            print(e)
            self.isLearning = False
            self.update_target()

    def save(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.online_network.eval()
