import random

import gym
import numpy as np
import torch
from scipy.misc import imresize


def preprocess(rgb):
    """
    rgb: height x width x channel(RGB)
    """
    # 210 x 160 x 3
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    x = 0.299 * r + 0.587 * g + 0.114 * b
    x = imresize(x, (110, 84)).astype(np.uint8)
    x = x[18:102]
    return x


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
