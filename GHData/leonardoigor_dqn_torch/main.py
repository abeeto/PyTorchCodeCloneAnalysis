import os
import torch
import numpy as np
from agent import DQNAgent
import gym

from teste import test, train


def set_seed(env, seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    env.seed(seed_value)
    env.action_space.np_random.seed(seed_value)


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

train_mode = True
env = gym.make('LunarLander-v2')
model_file = "./ll_online_net"
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space, device=device, epsilon_max=1.0,
    epsilon_min=0.01, epsilon_decay=0.99, memory_capacity=10000,
    discount=.99, lr=1e-3)
set_seed(env, 0)
if train_mode:
    train(env=env, agent=agent,
          train_eps=2000,
          memory_fill_eps=20,
          batch_size=64,
          update_freq=10,
          model_filename=model_file
          )
else:
    test(env=env, agent=agent, test_aps=5)
