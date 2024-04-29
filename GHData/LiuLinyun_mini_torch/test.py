import numpy as np 
from Var import Var
from Module import Module
from Optim import Optim
import Layer
import Fn as fn

import os
import random

class Net(Module):
    def __init__(self):
        super().__init__() 
        self.linear1 = Layer.Linear(1, 32)
        self.linear2 = Layer.Linear(32, 64) 
        self.linear3 = Layer.Linear(64, 64)
        self.linear4 = Layer.Linear(64, 1) 
        self.layers = [self.linear1, \
                        self.linear2, \
                        self.linear3, \
                        self.linear4]

    def forward(self, inputs):
        l0 = inputs[0]
        l1 = fn.sigmoid(self.linear1(l0))
        l2 = fn.sigmoid(self.linear2(l1))
        l3 = fn.sigmoid(self.linear3(l2))
        l4 = self.linear4(l3)
        return l4

net = Net()
batch_size = 100
optim = Optim(net, 0.001/batch_size)


for iter in range(1000):
    optim.zero_grad()
    loss = Var(np.array([0.0]))
    for i in range(batch_size):
        x = np.array([np.random.uniform(-1, 1)])
        y = x**2
        var_x, var_y = Var(x), Var(y)
        output = net([var_x])
        loss_ = fn.mse_loss(output, var_y)
        loss = fn.add(loss, loss_)
    print(loss.data/batch_size)
    loss.backward()
    optim.step()





