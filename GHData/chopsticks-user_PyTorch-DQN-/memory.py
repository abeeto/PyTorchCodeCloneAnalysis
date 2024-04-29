from collections import deque
from collections import namedtuple, deque
import random as R

transition_values = namedtuple("transition_values", ("current_state", "action", "next_state", "reward", "done"))

#reduce duplicate "memory" names
class memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory_buffer = deque([], maxlen = self.capacity)
        self.cur_mem_p = 0

    def update(self, *args):
        if self.cur_mem_p < self.capacity:
            self.memory_buffer.append(*args)
            self.cur_mem_p += 1
            return
        self.memory_buffer.popleft()
        self.memory_buffer.appendleft(*args)

    def sample(self, batch_size):
        return R.sample(range(batch_size), batch_size), R.sample(self.memory_buffer, batch_size)