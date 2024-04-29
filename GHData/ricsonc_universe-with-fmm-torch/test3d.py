from aljpy import arrdict, dotdict
import torch
import pybbfmm
import numpy as np
from ipdb import set_trace as st
from pybbfmm import orthantree, plotting
import matplotlib.pyplot as plt
from munch import Munch as M
from matplotlib.patches import Patch
from time import time

N = 1000000
D = 3
xs = np.random.uniform(size=(N, D))

prob = arrdict.arrdict(
    sources=xs,
    charges=np.ones(xs.shape[0]),
    targets=xs,
).map(lambda t: torch.as_tensor(t).float())

prob['kernel']=lambda a, b: 1.0/((a-b+1)**2).sum(-1)

t0 = time()
y = pybbfmm.solve(prob)
t1 = time()
print(t1-t0)

st()
