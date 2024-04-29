import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayBuffer(deque):

    def __init__(self, size):
        super().__init__(maxlen=size)

    def sample(self, batch_size):
        return random.sample(self, batch_size)

    def push(self, state, action, done, next_state, reward):
        t = Transition(state, action, done, next_state, reward)
        self.append(t)



