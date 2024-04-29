from aljpy import arrdict, dotdict
import torch
import pybbfmm
from ipdb import set_trace as st
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from munch import Munch as M
import imageio as ii
from time import time
import numpy as np
import sys

#150 and 160 both good..
x = np.load(f'out/{int(sys.argv[1]):04d}.npy')
# y = np.load('out/0160.npy')



# y[...,0] += 1.2
# xy = np.concatenate((x, y), axis = 0)
xy = x

# x = np.load('out/0320.npy')
# y = np.load('out/0330.npy')

# diff = x-y
# mag = np.minimum(np.abs(diff), 1-np.abs(diff))
# speed = np.sqrt((mag**2).sum(-1))

# plt.hist(speed, bins = 200)
# plt.show()
# st()

plt.scatter(xy[...,0], xy[...,1], s=0.1, alpha=0.5)
plt.axes().set_aspect('equal')
plt.axis('off')
plt.tight_layout()

plt.show()

