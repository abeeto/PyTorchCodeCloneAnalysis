import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DDQNModel(nn.Module):
    def __init__(self, num_actions=2):
        super(DDQNModel, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        val = self.relu(self.fc1_val(x))
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        adv = self.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)
        output = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return output

class DDQN():
    def __init__(self, n_action, lr=1e-6):
        self.criterion = nn.MSELoss()
        self.model = DDQNModel(n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, y_predict, y_target):
        """
        Update the weights of the DQN given a training sample
        @param y_predict:
        @param y_target:
        @return:
        """
        loss = self.criterion(y_predict, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay
        @param memory: a list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        @return: the loss
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*replay_data)

            state_batch = torch.cat(tuple(state for state in state_batch))
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            q_values_batch = self.predict(state_batch)
            q_values_next_batch = self.predict(next_state_batch)

            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

            action_batch = torch.from_numpy(
                np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))

            q_value = torch.sum(q_values_batch * action_batch, dim=1)

            td_targets = torch.cat(
                tuple(reward if terminal else reward + gamma * torch.max(prediction) for reward, terminal, prediction
                    in zip(reward_batch, done_batch, q_values_next_batch)))

            loss = self.update(q_value, td_targets)
            return loss