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
from scipy.stats import gaussian_kde as kde
from scipy.ndimage import gaussian_filter

x = np.load('out_slow/0150.npy')

N = 4096

xround = np.round(x * N).astype(np.int64) % N
xflat = xround[...,0] * N + xround[...,1]
xunique, counts = np.unique(xflat, return_counts = True)

canvas = np.zeros(N*N)
canvas[xunique] += counts
canvas = canvas.reshape(N, N)

canvas = gaussian_filter(canvas, sigma = 8, mode='wrap') + gaussian_filter(canvas, sigma = 4, mode='wrap')
canvas = np.log(canvas+0.1)

#canvas = np.clip(canvas, -1.2, 1.5)

# intensities = canvas.reshape(-1)
# plt.hist(intensities, bins=200)
# plt.show()

# plt.imshow(canvas, cmap='magma')
# plt.show()

canvas = plt.cm.magma(plt.Normalize(vmin=-1.2, vmax=1.5)(canvas))
ii.imsave('vis.png', canvas)
