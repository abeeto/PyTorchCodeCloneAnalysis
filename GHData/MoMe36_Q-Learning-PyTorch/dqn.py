import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


from common import ReplayBuffer, Env 
import numpy as np 
import gym 
from collections import deque
import random 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')


class DQN(nn.Module): 

	def __init__(self, sizes, hidden = 64): 

		nn.Module.__init__(self)

		self.sizes = sizes
		self.model = nn.Sequential(nn.Linear(sizes[0], hidden), nn.ReLU(), 
								   nn.Linear(hidden, hidden), nn.ReLU(), 
								   nn.Linear(hidden, sizes[1]))

		for module in self.model: 
			if isinstance(module, nn.Linear): 
				module.weight.data.uniform_(-0.01,0.01)
				module.bias.data.zero_()

	def forward(self, x): 

		output = self.model(x)
		return output

	def act(self, x, eps): 

		if np.random.random() < eps: 
			action = np.random.randint(0,2)
		else: 
			vals = self(x).detach()
			action = torch.max(vals, 1)[1].item()

		return action 

	def sample_action(self): 

		return torch.randint(low = 0, high = self.sizes[1], size= [1]).long().item()

	def compute_loss(self, memory, batch_size): 

		states, actions, rewards, next_obs, done = memory.sample(batch_size) 

		q_values = self(torch.tensor(states).float())
		selected_q_values = torch.gather(q_values, 1, torch.tensor(actions).long().reshape(-1,1))

		rewards = torch.tensor(rewards).reshape(-1,1).float()
		next_q_vals = self(torch.tensor(next_obs).float()).detach()
		max_q_vals = torch.max(next_q_vals, 1)[0].reshape(-1,1)
		masks = torch.tensor(done).float().reshape(-1,1)

		expected = rewards + 0.99*max_q_vals*masks
		loss = torch.mean(torch.pow(q_values - expected,2)) #F.smooth_l1_loss(selected_q_values, expected)

		return loss 



env = Env('CartPole-v0')
agent = DQN(env.sizes) 

adam = optim.Adam(agent.parameters(),1e-3)

memory = ReplayBuffer(1000)

epochs = 20000
batch_size = 32

max_eps = 1. 
min_eps = 0.01 
eps_decay = 8000 

eps = lambda max_eps, min_eps, eps_decay, epoch : min_eps + (max_eps - min_eps)*np.exp(-1.*epoch/eps_decay)
recap = []

episode_mean_reward = 0 

for episode in range(epochs): 

	s = env.reset()
	episode_reward = 0
	done = False
	
	while not done: 
	
		action = agent.act(s, eps(max_eps, min_eps, eps_decay, episode))

		ns, r, done, _ = env.step(action)

		memory.observe_episode(s, action, r, ns, done)

		
		s = ns
		episode_reward += r

		if done: 
			
			episode_mean_reward += episode_reward
			if(episode%100 == 0 and episode > 0): 
				print('Episode :{} Reward: {:.3f} Loss:{:.3f} Eps:{:.3f}'.format(episode, episode_mean_reward/100., loss.item(), eps(max_eps, min_eps, eps_decay, episode)))
				episode_mean_reward = 0 

			recap.append(episode_reward)
			episode_reward = 0 


			# TRAINING 
			if(len(memory) > batch_size): 
				loss = agent.compute_loss(memory, batch_size)
				adam.zero_grad()
				loss.backward()
				adam.step()


plt.plot(recap)

plt.show()




		





