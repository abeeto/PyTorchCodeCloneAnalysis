import numpy as np

class PPOmemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = [] # log_probs
        self.vals = [] # values that the critic calculate
        self.actions = [] 
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len((self.states))
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start] # list of batches with the indices... Not in order.
        
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs),\
               np.array(self.vals), \
               np.array(self.rewards),\
               np.array(self.dones), \
               batches
    
    def store_memory(self, state, action, probs, vals, reward, done): 
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)
    
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

