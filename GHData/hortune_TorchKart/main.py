import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from loader import TensorKartDataSet
import sys
import numpy as np



from dqn import DQN
from memory import ReplayBuffer
from utils import process_frame, discrete_action, actionTransform
from model import Network
from IPython import embed
from pretrain import preTrain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
arg = parser.parse_args()

if arg.train:
    model = Network().cuda()
    epochs = 150
    batch_size = 128
    preTrain(model, epochs, batch_size, diff=False, hist_eq=False, discrete=True)

import gym, gym_mupen64plus
env_list = [ 
    'Mario-Kart-Luigi-Raceway-v0',
    'Mario-Kart-Moo-Moo-Farm-v0',
    'Mario-Kart-Koopa-Troopa-Beach-v0',
    'Mario-Kart-Kalimari-Desert-v0',
    'Mario-Kart-Toads-Turnpike-v0',
    'Mario-Kart-Frappe-Snowland-v0',
    'Mario-Kart-Choco-Mountain-v0',
    'Mario-Kart-Mario-Raceway-v0',
    'Mario-Kart-Wario-Stadium-v0',
    'Mario-Kart-Sherbet-Land-v0',
    'Mario-Kart-Royal-Raceway-v0',
    'Mario-Kart-Bowsers-Castle-v0',
    'Mario-Kart-DKs-Jungle-Parkway-v0',
    'Mario-Kart-Yoshi-Valley-v0',
    'Mario-Kart-BansheeBoardwalk-v0',
    'Mario-Kart-Rainbow-Road-v0']
if arg.test:
    env = gym.make(env_list[0])
    dqn = DQN(env)
    dqn.test(diff=False)
