# By Oleksiy Grechnyev, IT-JIM, 30-Dec-2020
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch

########################################################################################################################
def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)


########################################################################################################################
def plot_data(x, y, zoom=1):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    plt.axis(np.array([-1.1, 1.1, -1.1, 1.1]) * zoom)
    plt.axis('off')

    m, c = 0, '.15'
    plt.axvline(0, ymin=m, c=c, lw=1, zorder=0)
    plt.axhline(0, xmin=m, c=c, lw=1, zorder=0)


########################################################################################################################
def plot_model(x, y, model):
    model.cpu()
    mesh = np.arange(-1.1, 1.1, 0.1)
    xx, yy = np.meshgrid(mesh, mesh)
    # print('xx=', xx)
    # print('yy=', yy)
    data0 = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T.astype('float32')
    with torch.no_grad():
        data = torch.from_numpy(data0)
        z = model(data).detach()
    z = z.argmax(axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(x, y)

########################################################################################################################
