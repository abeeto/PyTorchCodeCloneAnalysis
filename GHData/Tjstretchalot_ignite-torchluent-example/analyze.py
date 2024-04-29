"""Loads the model created by the runner and analyzes it using pca3dvis"""
import multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import torch
import torchvision
import mnist
from torchluent.fluent_module import StrippingModule
import numpy as np
import pca3dvis.pcs
import pca3dvis.worker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import operator
from functools import reduce


def _main():
    model = torch.load('out/mnist/model.pt')
    if isinstance(model, StrippingModule):
        model = model.child
    trainset = torchvision.datasets.MNIST(
        'datasets/mnist', transform=mnist.transform,
        train=True, download=True)
    # valset = torchvision.datasets.MNIST(
    #     'datasets/mnist', transform=transform,
    #     train=False, download=True)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=1)
    #val_loader = torch.utils.data.DataLoader(valset, batch_size=500, shuffle=True, num_workers=1)

    samples, labels = next(iter(train_loader))

    model.eval()
    with torch.no_grad():
        _, arr = model(samples)

    for i in range(len(arr)):
        arr[i] = arr[i].view(arr[i].shape[0], reduce(operator.mul, arr[i].shape[1:]))

    titles = [f'Layer {i}' for i in range(len(arr))]
    titles[0] = 'Input'
    titles[-1] = 'Output'

    markers = [
        (
            np.ones(samples.shape[0], dtype='bool'),
            {
                'c': labels.numpy(),
                'cmap': plt.get_cmap('Set1'),
                's': 20,
                'marker': 'o',
                'norm': mcolors.Normalize(labels.min().item(), labels.max().item())
            }
        )
    ]

    narr = [i.numpy() for i in arr]

    traj = pca3dvis.pcs.get_pc_trajectory(narr, labels.numpy())

    pca3dvis.worker.generate(
        traj, markers, titles, 'out/mnist/analyzed', True
    )

if __name__ == '__main__':
    _main()
