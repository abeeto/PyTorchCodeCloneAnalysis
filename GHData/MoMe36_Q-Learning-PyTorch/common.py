from collections import deque 
import random 
import gym 

import torch 

def discount_reward(rewards): 

	current = 0
	discounted = []

	for i in reversed(range(len(rewards))):

		current = current*0.99 + rewards[i]
		discounted.insert(0,current)

	return discounted	

class Env: 

	def __init__(self, env_id): 

		self.env_id = env_id
		self.env = gym.make(env_id)

	def to_tensor(self, x): 

		return torch.tensor(x).float().reshape(1,-1)

	def reset(self): 

		return self.to_tensor(self.env.reset())

	def step(self, action): 

		ns, r, done, _ = self.env.step(action)

		return self.to_tensor(ns), r, done, _

	@property
	def sizes(self):
		if self.env_id == 'CartPole-v0': 
			return [4,2]
		return [self.env.observation_space.shape[0], self.env.action_space.shape[0]]



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

        self.episode_states, self.episode_next_states, self.episode_rewards, self.episode_actions = [],[],[],[]
    
    def push(self, state, action, reward, next_state, done):
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), action, reward, torch.cat(next_state), done

    def observe_episode(self, s, a, r, ns, done): 

    	self.episode_states.append(s)
    	self.episode_actions.append(a)
    	self.episode_rewards.append(r)
    	self.episode_next_states.append(ns)

    	if done: 
    		self.compute_returns()

    def compute_returns(self):

    	discounted = discount_reward(self.episode_rewards)
    	done = [False for i in range(len(discounted)-1)]
    	done.append(True)

    	for s,a,r,ns,d in zip(self.episode_states, self.episode_actions, discounted, self.episode_next_states, done): 
    		self.push(s,a,r,ns,d)

    	self.episode_states, self.episode_next_states, self.episode_rewards, self.episode_actions = [],[],[],[]

    def __len__(self):
        return len(self.buffer)