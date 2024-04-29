import os
import time
import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import datetime
import random
from collections import OrderedDict
gui = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(4, 4))),
            ('act1', nn.Sigmoid()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(4, 4))),
            ('act2', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('act3', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(480, 128)),
            ('act4', nn.ReLU()),
            ('f7', nn.Linear(128, 4)),
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


cmd = 'python3 localrunner.py -p3 "python3 strat.py"  -p1 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot -p2 simple_bot'
cmd1 = 'python3 localrunner.py -p4 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p5 simple_bot -p6 simple_bot -p2 simple_bot'
cmd3 = 'python3 localrunner.py -p5 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p4 simple_bot -p6 simple_bot -p2 simple_bot'
cmd4 = 'python3 localrunner.py -p6 "python3 strat.py"  -p1 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p2 simple_bot'
cmd5 = 'python3 localrunner.py -p2 "python3 strat.py"  -p1 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot -p3 simple_bot'
cmd2 = 'python3 localrunner.py -p1 "python3 strat.py" -p2 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot'
cmds = [cmd2, cmd5, cmd, cmd1, cmd3, cmd4]

def generate_batch(batch_size):
    print(f'generating batch...')
    batch_actions, batch_states, batch_rewards = [], [], []

    for b in range(batch_size):
        choosen = random.choice(cmds)
        if gui:
            print(cmds.index(choosen)+1)
        os.system(choosen if gui else choosen+' --no-gui')
        time.sleep(0.1)
        os.system('find . -type f -name "*.log.gz" -exec rm -f {} \;')
        with open('log', 'r') as f:
            line = json.loads(f.readline().strip())
            batch_states.append(line)
            line = json.loads(f.readline().strip())
            batch_actions.append(line)
            line = json.loads(f.readline().strip())
            print(f'{line} - reward')
            batch_rewards.append(line)
        print(f'{b+1}/{batch_size} of batch ended')
    return batch_states, batch_actions, batch_rewards


def filter_batch(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])

    return elite_states, elite_actions


if __name__ == '__main__':
    batch_size = 100
    session_size = 100
    percentile = 90
    hidden_size = 200
    learning_rate = 0.0025
    completion_score = 200

    n_states = 961
    n_actions = 4

    # neural network
    net = Net()
    # load params
    try:
        net.load_state_dict(torch.load('params/param19'))
        net.eval()
    except FileNotFoundError:
        pass
    # loss function
    objective = nn.CrossEntropyLoss()
    # optimisation function
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    for i in range(session_size):
        # generate new sessions
        print(f'started session {i+1}')
        batch_states, batch_actions, batch_rewards = generate_batch(batch_size)
        elite_states, elite_actions = filter_batch(batch_states, batch_actions, batch_rewards, percentile)
        optimizer.zero_grad()

        tensor_states = torch.FloatTensor(np.array(elite_states).reshape([len(elite_states), 1, 31, 31]))
        tensor_actions = torch.LongTensor(elite_actions)

        action_scores_v = net(tensor_states)
        loss_v = objective(action_scores_v, tensor_actions)
        loss_v.backward()
        optimizer.step()

        # show results
        mean_reward = np.mean(batch_rewards),
        np.percentile(batch_rewards, percentile)
        print(f"{i}: loss={loss_v.item()}, reward_mean={mean_reward}")

        torch.save(net.state_dict(), 'params/param19')
        torch.save(net.state_dict(), 'params/param19' + f'-{datetime.datetime.now()}-reward{mean_reward}')

