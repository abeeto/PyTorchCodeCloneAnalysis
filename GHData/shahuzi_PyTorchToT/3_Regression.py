#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       3_Regression.py
@software:   pycharm
Created on   7/4/18 9:27 PM

"""

import torch as tc
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F


# unsqueeze : turn 1-dim data to 2-dim  (100) --> (100,1)
x = tc.unsqueeze(tc.linspace(-1,1,100),dim=1) # x data(tensor) shape = (100,1)
y = x.pow(2) + 0.2 * tc.rand(x.size())

# x,y = Variable(x),Variable(y)

plt.figure()
plt.scatter(x.numpy(),y.numpy())
plt.ion()
plt.show()


class Net(tc.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = tc.nn.Linear(n_feature,n_hidden) # hidden layer
        self.predict = tc.nn.Linear(n_hidden,n_output) # output layer

    def forward(self,input_data):  # feedforward
        fc1 = F.relu(self.hidden(input_data))
        pred = self.predict(fc1)    # in regression problem, we don't use activation function at the prediction layer,cause the output maybe -NaN to +NaN
        return pred



net = Net(n_feature=1,n_hidden=10,n_output=1)
print net

optimizer = tc.optim.SGD(net.parameters(),lr=0.5)
loss_fn = tc.nn.MSELoss()

for t in range(105):
    prediction = net(x)
    loss = loss_fn(prediction,y)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()        # backpropagation
    optimizer.step()       # apply gradients

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(),y.numpy())
        plt.plot(x.numpy(),prediction.detach().numpy(),'r-',lw=5)
        plt.text(0, 1, 'Step=%3d' % t, fontdict={'size': 10, 'color': 'red'})
        plt.text(0.5,0,'Loss=%.4f'%loss.item(),fontdict={'size':10,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
