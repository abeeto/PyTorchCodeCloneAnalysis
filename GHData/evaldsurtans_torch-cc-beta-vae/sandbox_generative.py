import argparse
import copy
import json
import math
import os
import pathlib
import random

import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy import spatial

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
import torchvision.utils
import torch.utils.data
import torch.distributions

import matplotlib.pyplot as plt
from torchvision import transforms

from modules.dict_to_obj import DictToObj
from modules_core.model_2 import Model



BATCH_SIZE = 36

dataset = torchvision.datasets.EMNIST(
    root='./tmp',
    split='balanced',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# examine samples to select idxes you would like to use


idx = 0
for x, y_idx in data_loader:
    break
    plt.rcParams["figure.figsize"] = (int(BATCH_SIZE/4), int(BATCH_SIZE/4))
    plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
    for i in range(BATCH_SIZE):
        plt.subplot(plt_rows, plt_rows, i + 1)
        plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))
        plt.title(f"idx: {idx}")
        idx += 1
        plt.tight_layout(pad=0.5)
    plt.show()
    if idx > 200:
        break


args = DictToObj(**{})
args.embedding_size = 32
model = Model(args)
model.load_state_dict(torch.load('./model_2_a.pt', map_location='cpu'))
model.eval()
torch.set_grad_enabled(False)


IDEXES_TO_ENCODE = [
   9, 167, 188, 212, 125
]

x_to_encode = []
for idx in IDEXES_TO_ENCODE:
    x_to_encode.append(dataset[idx][0])

plt_rows = int(np.ceil(np.sqrt(len(x_to_encode))))
for i in range(len(x_to_encode)):
    plt.subplot(plt_rows, plt_rows, i + 1)
    x = x_to_encode[i]
    plt.imshow(x[0].T, cmap=plt.get_cmap('Greys'))
    plt.title(f"idx: {IDEXES_TO_ENCODE[i]}")
    plt.tight_layout(pad=0.5)
plt.show()

t_x = torch.stack(x_to_encode, dim=0)
z, z_mu, z_sigma = model.encode_z(t_x)

z_mu = torch.mean(z, dim=0)
z_sigma = torch.std(z, dim=0)
#z_sigma = torch.ones_like(z_sigma) * 0.3

# z_mu = torch.mean(z_mu, dim=0)
# z_sigma = torch.mean(z_sigma, dim=0)

# z_mu: 0.7961747050285339 z_sigma: 0.3033522367477417
# z_mu: 0.7068101167678833 z_sigma: 0.6534739136695862
print(f'z_mu: {torch.std(z_mu).item()} z_sigma: {torch.std(z_sigma).item()}')

dist = torch.distributions.Normal(z_mu, z_sigma)
z_gen = [z_mu]
for i in range(BATCH_SIZE):
    z_gen.append(dist.sample())
t_z_gen = torch.stack(z_gen, dim=0)

t_x_gen = model.decode_z(t_z_gen)

plt.imshow(
    torchvision.utils.make_grid(t_x_gen)[0].T,
    cmap=plt.get_cmap('Greys')
)
plt.show()
