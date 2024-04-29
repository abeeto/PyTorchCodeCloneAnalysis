import gym
import torch
from torch.autograd import Variable

from model import DQNAgent
from env import Env
from configs import args
from utils import preprocess


env = Env(args)
legal_actions = env.action_space
agent = DQNAgent(env, legal_actions.n, args)
agent.dqn.load_state_dict(torch.load(args.checkpoint))
agent.train(False)

while True:
    observation = preprocess(env.reset())
    agent.reset_history()
    agent.push_observation(observation, repeat=args.history_size)
    done = False
    while not done:
        history = agent.get_history()
        action = agent.predict(history, test=0.05)
        print(action)
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        agent.push_observation(observation)
        env.render()
