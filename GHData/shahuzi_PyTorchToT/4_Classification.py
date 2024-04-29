#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       4_Classification.py
@software:   pycharm
Created on   7/5/18 1:24 PM

"""


import torch as tc
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F



class Net(tc.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = tc.nn.Linear(n_feature,n_hidden) # hidden layer
        self.predict = tc.nn.Linear(n_hidden,n_output) # output layer

    def forward(self,input_data):  # feedforward
        fc1 = F.relu(self.hidden(input_data))
        out = self.predict(fc1)    # in regression problem, we don't use activation function at the prediction layer,cause the output maybe -NaN to +NaN
        return out


# make fake data
n_data = tc.ones(100, 2)
x0 = tc.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = tc.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = tc.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = tc.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = tc.cat((x0, x1), 0).type(tc.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = tc.cat((y0, y1), ).type(tc.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# plt.figure()
# plt.scatter(x.numpy()[:,0],x.numpy()[:,1],c=y.numpy(),s=100,lw=0,cmap='RdYlGn')
# plt.show()

net = Net(2,10,2)   # output [0 1] or [1 0] one-hot label
print net

optimizer = tc.optim.SGD(net.parameters(),lr=0.01)
loss_fn = tc.nn.CrossEntropyLoss()



for t in range(105):
    out = net(x)
    loss = loss_fn(out,y)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()        # backpropagation
    optimizer.step()       # apply gradients

    if t % 2 == 0:
        plt.cla()
        prediction = tc.max(out,1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -3, 'Step=%3d' % t, fontdict={'size': 10, 'color': 'red'})
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()



