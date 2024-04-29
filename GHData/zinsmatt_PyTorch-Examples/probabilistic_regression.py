#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x) * np.exp(-(0.1*x)**2) #+ 0.1 * np.sin(2*x)
    # return (0.1*x)**2

NB_TRAIN_DATA = 100
XMIN_TRAIN = -10
XMAX_TRAIN = 10

X_train = np.linspace(XMIN_TRAIN, XMAX_TRAIN, NB_TRAIN_DATA).astype(np.float32)
Y_train = f(X_train)



def train_model(model, optimizer, loss_fn, X_batches, Y_batches, nb_epochs):
    model.train()
    for e in range(nb_epochs):
        losses = []
        for x, y in zip(X_batches, Y_batches):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Epoch %d mean loss = %f" % (e, np.mean(losses)))


def eval_model(model, X_data, Y_data):
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        data = torch.tensor(X_data).unsqueeze(1)
        output = model(data)
    y_pred = output.numpy()
    mean_error = np.sqrt(np.sum((y_pred - Y_data)**2)) / X_data.shape[0]
    plt.scatter(X_data, y_pred[:, 0], c="green", s=0.1)
    print("Mean error = ", mean_error)



def test_model(model, X_data):
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
         data = torch.tensor(X_test).unsqueeze(1)
         output = model(data)
         y_pred = output.numpy()
    plt.scatter(X_data, y_pred[:, 0], c="red", s=0.1)



class RegressionNet(nn.Module):
    def __init__(self, layers_size):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential()
        layers_size = [1] + layers_size
        for i in range(1, len(layers_size)):
            self.layers.add_module("linear_" + str(i), nn.Linear(layers_size[i-1],
                                                                 layers_size[i]))
            self.layers.add_module("relu_" + str(i), nn.ReLU())
        self.layers.add_module("final", nn.Linear(layers_size[-1], 1))

    def forward(self, x):
        return self.layers(x)


class ProbRegressionNet(nn.Module):    
    def __init__(self, layers_size):
        super(ProbRegressionNet, self).__init__()
        self.layers = nn.Sequential()
        layers_size = [1] + layers_size
        for i in range(1, len(layers_size)):
            self.layers.add_module("linear_" + str(i), nn.Linear(layers_size[i-1],
                                                                 layers_size[i]))
            self.layers.add_module("relu_" + str(i), nn.ReLU())
        self.layers.add_module("final", nn.Linear(layers_size[-1], 2))

    def forward(self, x):
        return self.layers(x)
    
def AleatoricUncertaintyLoss(x, y):
    pred = x[:, 0].view((-1, 1))
    uncertainty = x[:, 1].view((-1, 1))
    si = uncertainty**2
    
    # u2 = uncertainty**2
    # si = torch.log(u2)
    
    N = x.shape[0]
    
    # loss = (1.0 / N) * torch.sum((1.0 / (2.0 * u2)) * (y - pred)**2 + 0.5 * torch.log(u2))
    # loss = torch.mean((pred - y)**2)
    # loss = (1.0 / N) * torch.sum(0.5 * torch.exp(-si) * (y - pred)**2 + 0.5 * si)
    
    loss = torch.sum(torch.log(si) + (y - pred)**2 / si)
    
    
    return loss 

    

NB_EPOCHS = 2000
LEARNING_RATE = 0.0001
GAMMA= 0.1
BATCH_SIZE = 64

device = torch.device("cpu")

model = RegressionNet([32, 64, 128, 64, 32]).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

X_tensor = torch.Tensor(X_train).unsqueeze(1).to(device)
Y_tensor = torch.Tensor(Y_train).unsqueeze(1).to(device)

X_batches = X_tensor.split(BATCH_SIZE)
Y_batches = Y_tensor.split(BATCH_SIZE)

# train_model(model, optimizer, nn.MSELoss(), X_batches, Y_batches, NB_EPOCHS)


model2 = ProbRegressionNet([32, 64, 128, 128, 64, 32]).to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
scheduler = StepLR(optimizer2, step_size=1, gamma=GAMMA)

train_model(model2, optimizer2, AleatoricUncertaintyLoss, X_batches, Y_batches, NB_EPOCHS)


#%% EVAL

plt.figure("Training data")
plt.scatter(X_train, Y_train, s=0.1)


ediff = X_train[1] - X_train[0]
X_eval = X_train + ediff / 2
X_eval = X_eval[:-1]
Y_eval = f(X_eval)

# eval_model(model, X_eval, Y_eval)

#%% TEST
X_test = np.linspace(XMIN_TRAIN-10, XMAX_TRAIN+10, 1100, dtype=np.float32)
test_model(model2, X_test)

#%%
model2.to(torch.device("cpu"))
model2.eval()
with torch.no_grad():
     data = torch.tensor(X_test).unsqueeze(1)
     output = model2(data)
     y_pred = output.numpy()
    
y_pred_mean = y_pred[:, 0]
# uncertainty = np.sqrt(np.exp(y_pred[:, 1]))
uncertainty = y_pred[:, 1]
y_pred_1 = y_pred_mean + uncertainty
y_pred_2= y_pred_mean - uncertainty

plt.scatter(X_test, y_pred_mean, c="red", s=0.1)
plt.scatter(X_test, y_pred_1, c="orange", s=0.1)
plt.scatter(X_test, y_pred_2, c="orange", s=0.1)
