from collections import namedtuple
import random
import numpy as np
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def __len__(self):
        return len(self.memory)
    
    def sample_by_idx(self, idxes):
        transitions = []
        for i in idxes:
            transitions.append(self.memory[i])
        return Transition(*zip(*transitions))
        
