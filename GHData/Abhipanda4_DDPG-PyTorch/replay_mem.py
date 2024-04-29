import numpy as np

class ReplayMemory:
    def __init__(self, max_capacity, state_dim, n_actions):
        self.capacity = max_capacity
        self.index = None
        self.index_list = list(range(max_capacity))

        self.S       = np.zeros((max_capacity, state_dim))
        self.S_prime = np.zeros((max_capacity, state_dim))
        self.actions = np.zeros((max_capacity, n_actions))
        self.rewards = np.zeros((max_capacity, 1))
        self.is_done = np.full((max_capacity, 1), True, dtype=bool)

    def add_to_memory(self, experience):
        '''
        Puts experience into memory buffer
        args:
            :experience: a tuple consisting of (S, A, S_prime, R, is_done)
        '''
        if self.index == None or self.index == self.capacity - 1:
            self.index = 0
        else:
            self.index += 1

        S, A, S_prime, R, is_done = experience
        self.S[self.index]       = S
        self.S_prime[self.index] = S_prime
        self.actions[self.index] = A
        self.rewards[self.index] = R
        self.is_done[self.index] = is_done

    def sample(self, batch_size):
        '''
        Randomly sample from buffer.
        args:
            :batch_size: number of experiences to sample in a minibatch
        '''
        indices = np.random.choice(self.index_list, batch_size)
        S       = self.S[indices]
        A       = self.actions[indices]
        S_prime = self.S_prime[indices]
        R       = self.rewards[indices]
        is_done = self.is_done[indices]

        return (S, A, S_prime, R, is_done)
