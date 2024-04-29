import random
from collections import deque
from typing import List, Tuple, Dict
from multiprocessing import Process

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import gym
import matplotlib.pyplot as plt

from lib import Config, DQNTrainer, plot_dqn_train_result


def train_dqn(config, steps=50000):
    trainer = DQNTrainer(config)
    result = trainer.train(steps)

    return result


config = Config(run_name="Plain", env_id="CartPole-v1", device="cuda", n_steps=8, verbose=True)
train_result = train_dqn(config)

plot_dqn_train_result(train_result, label="4-Step DQN", alpha=0.9)
plt.axhline(y=500, color='grey', linestyle='-')  # 500 is the maximum score!
plt.xlabel("steps")
plt.ylabel("Episode Reward")

plt.legend()
plt.title("Training Comparison between Techniques for DQN")
plt.show()
