import pdb
import gym
import go2goal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as Models
import numpy as np
from main import *


np.set_printoptions(suppress=True, linewidth=300, precision=4,
                    formatter={'float_kind':'{:10.6f}'.format})



if __name__ == '__main__':
    N_EPS = 12
    env = gym.make('Go2Goal-v0', config={'num_iter': 1, 'dt': 0.1})
    policy = Models.MBRLPolicy(50, 0.1)
    policy.load_state_dict(torch.load('/home/aarg/Documents/mbrl_torch_g2g/models/pi30082019_2303'))
    evaluate(env, policy, N_EPS)
    env.close()
    # torch.save(policy.state_dict(), f'pi{time.strftime("%d%m%Y_%H%M", time.localtime())}')