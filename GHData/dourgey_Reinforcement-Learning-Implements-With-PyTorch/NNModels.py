import torch
import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, space_n, n_action, use_dueling=False, device="cpu"):
        super(DQNNet, self).__init__()
        self.use_dueling = use_dueling

        self.fc1 = nn.Linear(space_n, 128)
        self.fc2 = nn.Linear(128, 256)

        if self.use_dueling:
            self.v = nn.Linear(256, 1)

        self.action = nn.Linear(256, n_action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if self.use_dueling:
            q = self.action(x)
            v = self.v(x)
            action = v + (q - torch.mean(q))
        else:
            action = self.action(x)

        return action

class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        # # start to do the init...
        # nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # # init the bias...
        # nn.init.constant_(self.conv1.bias.data, 0)
        # nn.init.constant_(self.conv2.bias.data, 0)
        # nn.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x.transpose_(3, 2).transpose_(2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1, 32 * 7 * 7)

        return x

class DeepMindDQNNet(nn.Module):
    def __init__(self, space_n, n_actions, use_dueling=False):
        super(DeepMindDQNNet, self).__init__()
        self.use_dueling = use_dueling
        self.conv = deepmind()
        self.fc = nn.Linear(32 * 7 * 7, 256)
        self.action = nn.Linear(256, n_actions)
        if use_dueling:
            self.state_value = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.relu(self.fc(x))
        action_value = self.action(x)
        if self.use_dueling:
            state_value = self.state_value(x)
            action_value = state_value + (action_value - torch.mean(action_value))
        return action_value