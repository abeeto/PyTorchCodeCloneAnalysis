import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from model import QNetwork
from collections import deque, namedtuple

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment"""

    def __init__(self, action_size):
        """Initialize an Agent object.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action"""

        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, stacked_state, action, reward, stacked_next_state, done):
        # Save experience memory in replay memory
        self.memory.add(stacked_state, action, reward, stacked_next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # After enough samples are available in the memory, get a random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, stacked_state, epsilon):
        """Returns actions for given state as per current policy

        Params:
            state(array_like): current state
            eps(float): epsilon, for epsilon-greedy action selection"""

        # stacked_state = torch.from_numpy(stacked_state).float().unsqueeze(0).to(device)
        stacked_state = torch.from_numpy(stacked_state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(stacked_state)
        self.qnetwork_local.train()

        # Epsilon-greedy policy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s,a,r,s',done)
            gamma (float): discount factor"""

        stacked_states, actions, rewards, stacked_next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(stacked_next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(stacked_next_states)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(stacked_states).gather(1, actions)
        Q_expected = self.qnetwork_local(stacked_states)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the los
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        w_target = tau*w_local + (1-tau)*w_target

        Params:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter"""

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed=3):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["stacked_state", "action",
                                                                "reward", "stacked_next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, stacked_state, action, reward, stacked_next_state, done):
        """Add a new experience to memory"""
        e = self.experience(stacked_state, action, reward, stacked_next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        stacked_states = torch.from_numpy(np.vstack([e.stacked_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        stacked_next_states = torch.from_numpy(np.vstack([e.stacked_next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (stacked_states, actions, rewards, stacked_next_states, dones)

    def __len__(self):
        """Return current size of internal memory"""
        return len(self.memory)

