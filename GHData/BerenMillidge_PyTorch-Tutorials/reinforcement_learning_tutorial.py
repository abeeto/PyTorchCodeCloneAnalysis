# okay, the iam here is to do the reinforcement learning tutorial, which is also 
# the first thing I#m doing using gym. this is cool. it's so cool that ai results 
# are so reproducible and easily integrated into tutorials... that is just relaly cool
#and hopefully this will generally allow me to actually start doing proper ml, only a year into my phd!

import gym
import math 
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# create the environment
env = gym.make('CartPole-v0').unwrapped
#setup matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from Ipython import display

plt.ion()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# so, use experience rplay memory in order to train fro ma batch over and over again
# and sampling from it randomyl decorrelates batch samples ,which leads to more stable training

# so create two classes - transition - a named tuple representing a single environmental transition
# replay memory - a cyclic buffer of bounded size that holds recently observec transitions#
# and a sample method selecting a random batch for training

Transition = namedtuple('Transition',('state', 'action','next_state','reward'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory =[]
		self.position = 0

	def push(self, *args):
		#saves a transition
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position=(self.position + 1) % self.capcity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size) # use this function in my dataset for random sampling!

	def __len__(self):
		return len(self.memory)

	# huber loss is an interesting type of loss which acts like mean squared when error is small
	# but like mean absolute when it is large... so that is interesting
	# and more robust when it is noisy!

# 
# so will model Q with a CNN which takes the image patches and figure its out
# it will take in only the difference(!) between thetwo creen patches
# and has two outputs, being Q(left) and Q(right)
# i.e. the quality of each action given the current input... all of this is cool!

class DQN(nn.Module):

	def __init__(self):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3,16, kernel_size=5, strides=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2(16,32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.head = nn.Linear(448,2)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

# okay, so a straightforward convnet for the DQN
# next is the input extraction for figuring out how to get this to work

resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

#code from gym
screen_width = 600

def get_cart_location():
	world_width = env.x_threshold * 2 
	scale = screen_width / world_width
	return int(env.state[0] * scale + screen_width / 2.0) # middle of cart

def get_screen():
	screen = env.render(mode='rgb_array').transpose((2,0,1)) # transpose into torch order - chw instead of tf order hwch
	#strip off top and bottom of the screen
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = get_cart_location()
	if cart_location < view_width // 2:
		slice_rance = slice(view_width)
	elif cart_location > (screen_width - view_width //2):
		slice_rance = slice(-view_width, None)
	else:
		slice_rance = slice(cart_location - view_width // 2,
			cart_location + view_width //2)
	# I really don't udnerstand what on earth is going on here!
	# it's probably smiple but  don't understand it!
	screen = screen[:,:, slice_rance]
	# convert to float, rescale and conveert to torch tensor
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
	screen = torch.from_numpy(screen)
	#reisze and add a batch dimension
	return resize(screen).unsqueeze(0).to(device)

"""
env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1,2,0).numpy(), interpolation='none') # god knows what thi sdoes
plt.title('Example extracted screen')
plt.show()
"""
# it's not working, and I'm nottotally sure why... argh... I guess I don't need to show it!

# so, some smiple functions and hyperparaets to figure it out!

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# define networks
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict()) # I'm not sure what the difference is between the policy net
# and the target net
optim = RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
	global steps_done # this eems weird
	sample =random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \ math.exp(-1. * steps_done / EPS_DECAY) # I assume this is some wierd kind of lambda?
	steps_done +=1
	if sample > eps_threshold:
		with torch.no_grad():
			return policy_net(state).max(1)[1].view(1,1) # not totally sure what thi smeans
			# I dno't have a good mental model of the return values of these objects yey
	else:
		return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	if len(durations_t) >=100:
		means = duratoins_t.unfold(0,100,1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
	plt.pause(0.001)
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())
		# not sure what this is at all... but who knows?

# finally the code for training the model is needed
# this is a pretty huge function to be honest!
# I don't know the difference between the policy network and the target network and this is difficult

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)

	batch = Transition(*zip(*transitions)) # I don't know what this means?

	# compute mask of non
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # god knows what any of this means?
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # I'm not suer I'm learning torch to any reasonable level like this?

# now it is the main training loop - reset environment and initialie state tensor, then sample an action
# execute it, observe the next screen and the reward, and optimize ou model again!
num_episodes = 50
for i_eposode in range(num_episodes):
	env.reset() # reset environment
	last_screen = get_screen()
	current_screen = get_screen() # not sure these are the same
	state = current_screen - last_screen
	for t in count(): # what on earth is this
	action = select_action(state)
	_, reward, done, _ = env.step(action.item())#this uses the gym api I have no idea
	reward = torch.tensor([reward], device=device)

	#observe new state
	last_screen = current_screen
	current_screen = get_screen()
	if not done:
		next_state = current_screen - last_screen
	else:
		next_state =None

	# store the transision
	memory.push(state, action, next_state, reward)
	state = next_state

	# optimize model
	optimize_model()
	if done:
		episode_durations.append(t+1)
		plot_durations()
		break

	# update target network
	if i_eposode % TARGET_UPDATE == 0:
		target_net.load_state_dict(policy_net.steps_dict)

print("Complete")
env.render()
env.close()
plt.ioff()
plt.show()