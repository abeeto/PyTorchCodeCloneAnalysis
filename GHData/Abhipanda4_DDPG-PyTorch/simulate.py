import torch
from torch.autograd import Variable
import gym
from model import *
from environment import *

env = Game()
agent = Actor(env.state_dim, env.n_actions, env.limit)

try:
    agent.load_state_dict(torch.load("./saved_models/best_agent.pth"))
except:
    print("No pretrained model found, using random model!!")
    pass

is_done = False
S = env.reset()

while not is_done:
    S = Variable(torch.FloatTensor(S))
    A = agent(S).data.numpy()
    S, R, is_done = env.take_action(A)
    env.env.render()
