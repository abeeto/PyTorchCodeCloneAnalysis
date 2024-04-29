import torch
import gym
import numpy as np
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


env = gym.make('Pendulum-v0')

s_dim = env.observation_space.shape[0]
a_max = env.action_space.high[0]
a_min = env.action_space.low[0]
tau = 0.001
max_episodes = 200
batch_size = 128
gamma = 0.99
EPS = 0.003


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def lens(self):
		return self.len

	def add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		transition = (s,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)


def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)



class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,256)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = torch.Tensor([action_lim])

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = F.tanh(self.fc4(x))

		action = action * self.action_lim

		return action




# class Critic(nn.Module):

# 	def __init__(self, state_dim, action_dim):
# 		"""
# 		:param state_dim: Dimension of input state (int)
# 		:param action_dim: Dimension of input action (int)
# 		:return:
# 		"""
# 		super(Critic, self).__init__()

# 		self.state_dim = state_dim
# 		self.action_dim = action_dim

# 		self.fcs1 = nn.Linear(state_dim,400)
# 		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
# 		self.fcs2 = nn.Linear(400,300)
# 		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

# 		self.fca1 = nn.Linear(action_dim,300)
# 		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

# 		self.fc1 = nn.Linear(300,1)
# 		self.fc1.weight.data.uniform_(-EPS,EPS)

# 	def forward(self, state, action):
# 		"""
# 		returns Value function Q(s,a) obtained from critic network
# 		:param state: Input state (Torch Variable : [n,state_dim] )
# 		:param action: Input ACriticction (Torch Variable : [n,action_dim] )
# 		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
# 		"""
# 		s1 = F.relu(self.fcs1(state))
# 		s2 = F.relu(self.fcs2(s1))
# 		a1 = F.relu(self.fca1(action))
# 		x = s2 + a1

		
# 		x = self.fc1(x)

# 		return x


# class Actor(nn.Module):

# 	def __init__(self, state_dim, action_dim, action_lim):
# 		"""
# 		:param state_dim: Dimension of input state (int)
# 		:param action_dim: Dimension of output action (int)
# 		:param action_lim: Used to limit action in [-action_lim,action_lim]
# 		:return:
# 		"""
# 		super(Actor, self).__init__()

# 		self.state_dim = state_dim
# 		self.action_dim = action_dim
# 		self.action_lim = torch.Tensor([action_lim])

# 		self.fc1 = nn.Linear(state_dim,400)
# 		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

# 		self.fc2 = nn.Linear(400,300)
# 		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

# 		self.fc4 = nn.Linear(300,action_dim)
# 		self.fc4.weight.data.uniform_(-EPS,EPS)

# 	def forward(self, state):
# 		"""
# 		returns policy function Pi(s) obtained from actor network
# 		this function is a gaussian prob distribution for all actions
# 		with mean lying in (-1,1) and sigma lying in (0,1)
# 		The sampled action can , then later be rescaled
# 		:param state: Input state (Torch Variable : [n,state_dim] )
# 		:return: Output action (Torch Variable: [n,action_dim] )
# 		"""
# 		x = F.relu(self.fc1(state))
# 		x = F.relu(self.fc2(x))
# 		action = F.tanh(self.fc4(x))

# 		action = action * self.action_lim

# 		return action



def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


class DDPG():
    def __init__(self):
        self.actor = Actor(s_dim,1,a_max)
        self.critic = Critic(s_dim,1)
        self.target_actor = Actor(s_dim,1,a_max)
        self.target_critic = Critic(s_dim,1)
        hard_update(self.target_actor,self.actor)
        hard_update(self.target_critic,self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),1e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),1e-4)
        self.noise = OrnsteinUhlenbeckActionNoise(1)
        self.buffer = MemoryBuffer(int(1e6))
        # if torch.cuda.is_available():
            # self.actor.cuda()
            # self.

    def get_action(self,state,noise=True):
        state = V(torch.Tensor(state))
        action = self.actor.forward(state).detach()
        if noise:
            new_action = action.data.numpy() + (self.noise.sample() * a_max)
        else:
            new_action = action.data.numpy()
        return new_action


    def evaluate(self):
        for ep in range(max_episodes):
            obs = env.reset()
            print("episode",ep)
            done = False
            Reward = 0
            while not done:
                if ep > 70:
                    env.render()
                state = np.float32(obs)
                action = self.get_action(state)
                new_obs, r, done, info = env.step(action)
                Reward += r
                d = np.reshape(np.array(1.0 if done == True else 0.0, dtype=np.float32),(1,1))
                self.buffer.add(obs,action,r,new_obs)
                obs = new_obs
                self.experience_replay()
            print("Reward:",Reward)

    def experience_replay(self):
        s1,a1,r1,s2 = self.buffer.sample(batch_size)
        s1 = V(torch.Tensor(s1))
        a1 = V(torch.Tensor(a1))
        r1 = V(torch.Tensor(r1))
        s2 = V(torch.Tensor(s2))

        #update critic
        a2 = self.target_actor.forward(s2).detach()
        next_q = torch.squeeze(self.target_critic.forward(s2,a2).detach())
        y_expected = r1 + gamma * next_q
        y_predicted = torch.squeeze(self.critic.forward(s1,a1))
        loss_c = F.smooth_l1_loss(y_predicted,y_expected)
        self.critic_optimizer.zero_grad()
        loss_c.backward()
        self.critic_optimizer.step()
        #update actor
        a = self.actor.forward(s1)
        loss_a = -1*torch.sum(self.critic.forward(s1,a))
        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor,self.actor,tau)
        soft_update(self.target_critic,self.critic,tau)

if __name__ == '__main__':
    ddpg = DDPG()
    ddpg.evaluate()