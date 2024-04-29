# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

import matplotlib.pyplot as plt
import pandas
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from seq2seq import Seq2SeqModel


class FourierModel(nn.Module):

    def reinit(self):

        size = self.size
        dtype = torch.float

        coef = 0.1

        with torch.no_grad():

          self.linear1 += coef * torch.randn(1, size, dtype=dtype)
          self.linear2 += coef * torch.randn(1, size, dtype=dtype)
          self.b1 += coef * torch.randn(size, dtype=dtype)
          self.b2 += coef * torch.randn(size, dtype=dtype)
          self.c1 += coef * torch.randn(size, 1, dtype=dtype)
          self.c2 += coef * torch.randn(size, 1, dtype=dtype)
          self.c += coef * torch.randn(1, 1, dtype=dtype)

          self.linear3 += coef * torch.randn(1, 10, dtype=dtype)
          self.b3 += coef * torch.randn(10, dtype=dtype)
          self.c3 += coef * torch.randn(10, 1, dtype=dtype)


    def __init__(self, size):
        super(FourierModel, self).__init__()
        self.size = size
        dtype = torch.float

        self.linear1 = torch.randn(1, size, dtype=dtype, requires_grad=True)
        self.linear2 = torch.randn(1, size, dtype=dtype, requires_grad=True)
        self.b1 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.b2 = torch.randn(size, dtype=dtype, requires_grad=True)
        self.c1 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c2 = torch.randn(size, 1, dtype=dtype, requires_grad=True)
        self.c = torch.randn(1, 1, dtype=dtype, requires_grad=True)

        self.linear3 = torch.randn(1, 10, dtype=dtype, requires_grad=True)
        self.b3 = torch.randn(10, dtype=dtype, requires_grad=True)
        self.c3 = torch.randn(10, 1, dtype=dtype, requires_grad=True)

        self.linear4 = torch.ones(1, 10, dtype=dtype, requires_grad=True)
        self.linear5 = torch.ones(1, 10, dtype=dtype, requires_grad=True)
        self.b4 = torch.randn(10, dtype=dtype, requires_grad=True)
        self.c4 = torch.randn(10, 1, dtype=dtype, requires_grad=True)



        self.optimizer1 = torch.optim.Adagrad(
            [self.linear2, self.linear1, self.linear3, self.linear4, self.linear5, self.b1, self.b2, self.b3, self.b4],
            lr=0.2)
        self.optimizer2 = torch.optim.Adagrad(
            [self.c, self.c1, self.c2, self.c3, self.c4],
            lr=0.4)

        self.loss_fn = torch.nn.MSELoss(reduction="sum")

    def weighted_mse(self, y1, y2, weights):
        lseq = (y1 - y2) ** 2
        return lseq.view(len(y1)).dot(weights)

    def forward(self, x):
        y = torch.cos(x.mm(self.linear1) + self.b1).mm(self.c1) + \
            torch.relu(torch.cos(x.mm(self.linear3) + self.b3)).mm(self.c3) + \
            torch.sin(x.mm(self.linear2) + self.b2).mm(self.c2) + \
            self.c

        return y


    def train(self, data, steps, weights=None):

        loss = 0
        y_pred = None

        n = len(data)

        x, dx = np.linspace(0, n, n, endpoint=False, retstep=True)

        if weights is None:
            weights = torch.ones(len(x), dtype=torch.float)
        else:
            weights = torch.tensor(weights, dtype=torch.float)

        x = torch.tensor(x, dtype=torch.float).reshape(len(x), 1)
        y = torch.tensor(data, dtype=torch.float).reshape(len(x), 1)

        all_linear1_params = torch.cat([p.view(-1) for p in [self.c1, self.c2]])

        for t in range(steps):

            y_pred = self.forward(x)
            l1_regularization = 0.01 * torch.norm(all_linear1_params, 1)
            # loss = self.loss_fn(y_pred, y)
            loss = self.weighted_mse(y_pred, y, weights)
            l1loss = loss + l1_regularization

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            l1loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

        return loss, y_pred



def rolling_window(a, window, step_size=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def LRT(y, h, model):

    res = []

    weights = np.random.poisson(1.0, len(y))
    # weights = np.ones(len(y))
    weights_slides = rolling_window(weights, 2*h)
    y_slides = rolling_window(y, 2*h)

    for i in range(10):
       model.train(y_slides[i], 500, weights_slides[i])

    for (y12, w12) in zip(y_slides, weights_slides):

        y1 = y12[0:h]
        y2 = y12[h:2*h]

        w1 = w12[0:h]
        w2 = w12[h:2 * h]

        # model.reinit()

        loss1, y_pred = model.train(y1, 100, w1)
        loss2, y_pred = model.train(y2, 100, w2)
        loss12, y_pred = model.train(y12, 100, w12)

        lrt = (loss12 - loss1 - loss2).detach().numpy()
        res.append(lrt)

        print(lrt)

    return res



dataframe = pandas.read_csv('data/ptbdb_normal.csv', engine='python').values

dataframe1 = pandas.read_csv('data/ptbdb_abnormal.csv', engine='python').values

row = dataframe[1, 10:150]
row1 = dataframe1[10, 10:150]
row2 = dataframe1[12, 10:150]
model = Seq2SeqModel(50)


data = np.concatenate([row, row, row, row, row, row1, row])

plt.plot(data)
plt.show()

#lrt_res = LRT(data, 2 * len(row), model)
# plt.plot(lrt_res)
# plt.show()

loss, y_pred = model.train(np.concatenate([row, row, row, row]), 50)

print(loss)


plt.plot(y_pred.detach().numpy())
plt.show()