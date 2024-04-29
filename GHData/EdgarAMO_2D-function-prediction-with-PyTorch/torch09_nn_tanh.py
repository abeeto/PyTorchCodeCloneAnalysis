# Date:     09 / 05 / 2021
# Author:   Edgar Martinez-Ojeda
# Program:  nn non-linear net

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

# 2D function:
def fun(max_lim=6.0, min_lim=-6.0, n=100):
    x = np.linspace(min_lim, max_lim, n)
    y = np.linspace(min_lim, max_lim, n)
    x_, y_ = np.meshgrid(x, y)

    A = np.hstack((x_.flatten().reshape(-1, 1), y_.flatten().reshape(-1, 1)))
    b = np.sin(x_) + np.cos(y_) # <----------------- change this as you wish!
    b = b.flatten()

    return A, b.reshape(-1, 1)

# plot contours:
def plot_data(arr, tmax, tmin):
    """ contour plot """
    plt.style.use('classic')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(arr,
                    origin='lower',
                    interpolation=None,
                    extent=(tmin, tmax, tmin, tmax),
                    cmap='jet')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('z=sin(x)+cos(y)')
    cbar = fig.colorbar(ax=ax, mappable=img, orientation='vertical')
    plt.show()

# training loop:
def train(X_train, y_train, X_test, y_test, net):
    """ loop with training and validation set """
    
    # convert X, y to tensors:
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # iterator:
    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)

    # optimizer:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()

    # loss accumulator:
    time_line = []
    train_metric = []
    test_metric = []

    # loop:
    for epoch in range(epochs):
        # update parameters:
        for Xb, yb in train_loader:
            train_ls = loss(net(Xb), yb)
            optimizer.zero_grad()
            train_ls.backward()
            optimizer.step()
        # update train and test losses:
        with torch.no_grad():
            if not epoch % 50:
                time_line.append(epoch)
                metric = 0
                for Xb, yb in train_loader:
                    metric += loss(net(Xb), yb) / batch_size
                train_metric.append(metric)
                metric = 0
                for Xb, yb in test_loader:
                    metric += loss(net(Xb), yb) / batch_size
                test_metric.append(metric)
                # verbose:
                print('Epoch: ', epoch)

    # final report of the losses:         
    print('Train loss.....{0:6.3f}'.format(train_metric[-1]))
    print('Test loss......{0:6.3f}'.format(test_metric[-1]))

    # plot losses with respect to epochs:
    plt.plot(time_line, train_metric, color='b')
    plt.plot(time_line, test_metric, color='r')
    plt.show()
              
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

if __name__ == '__main__':
    # build a synthetic data set:
    n = 50
    sample = n ** 2 
    max_lim = 6.00
    min_lim = -6.0
    X, y = fun(max_lim, min_lim, n)

    # train-test split:
    ftrain = 0.8                    # train set fraction
    ftest = 1 - ftrain              # test set fraction

    ntrain = round(sample * ftrain) # train set size
    ntest = round(sample * ftest)   # test set size
    
    train_ix, test_ix = random_split(range(sample), [ntrain, ntest])
    X_train, y_train = X[train_ix, :], y[train_ix]
    X_test, y_test = X[test_ix, :], y[test_ix]
    
    # global hyper parameters:
    lr = 5e-3
    epochs = 500
    batch_size = 20

    # net, weights and bias:
    net = nn.Sequential(
        nn.Linear(2, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 1))

    # train the net:
    train(X_train, y_train, X_test, y_test, net)

    # test the net:
    def test_net(tmax, tmin, tn):
        Xt, _ = fun(max_lim=tmax, min_lim=tmin, n=tn)
        Xt = torch.tensor(Xt, dtype=torch.float32)
        yt = net(Xt)
        yt = yt.reshape(tn, tn)
        yt = yt.detach().numpy()
        plot_data(yt, tmax, tmin)

    # plot original data set:
    def plot_synthetic_data():
        global y
        y = y.reshape(n, n)
        plot_data(y, max_lim, min_lim)
