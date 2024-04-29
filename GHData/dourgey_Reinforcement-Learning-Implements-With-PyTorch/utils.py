import torch
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQNReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ActionSelector(object):
    def __init__(self):
        pass

    def select(self, agent, state):
        pass

class EGreedActionSelector(ActionSelector):
    def __init__(self, epsilon_begin=1, epsilon_end=0.01):
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon = self.epsilon_begin

    def select(self, policy_net, state, n_actions):
        self.epsilon_decay()
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                a = policy_net(state)
                return a.max(1)[1].unsqueeze(0)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


    def epsilon_decay(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * 0.99)