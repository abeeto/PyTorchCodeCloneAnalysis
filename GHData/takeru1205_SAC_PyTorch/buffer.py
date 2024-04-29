from collections import deque
class ReplayBuffer:
    def __init__(self, max_size=int(1e5)):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

