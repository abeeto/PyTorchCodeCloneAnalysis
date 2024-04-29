#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       6_save_restore.py
@software:   pycharm
Created on   7/5/18 2:16 PM

"""

import torch as tc
import matplotlib.pyplot as plt
# fake data
x = tc.unsqueeze(tc.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*tc.rand(x.size())  # noisy y data (tensor), shape=(100, 1)



def save_net(in_net,filename,entire=True):
    """
        save network
        if entire=True(default) save the entire network
        else just save parameters

    """
    if entire:
        tc.save(in_net,filename)
    else:
        tc.save(in_net.state_dict(),filename)
    print 'Save as %s --- Done'%(filename)

def resotre_net(filename,entire=True):
    """
    :param filename: the network filename
    :param entire: if True,will reload the entire network else just reload parameters,in this case,we should bulid a network framework the same
                   as the network to be load
    :return: net
    """
    if entire:
        net = tc.load(filename)
    else:
        net = tc.nn.Sequential(
            tc.nn.Linear(1, 10),
            tc.nn.ReLU(),
            tc.nn.Linear(10, 1)
        )
        net.load_state_dict(tc.load(filename))
    print 'Reload %s --- Done'%(filename)
    return net



net = tc.nn.Sequential(
    tc.nn.Linear(1,10),
    tc.nn.ReLU(),
    tc.nn.Linear(10,1)
)

optimizer = tc.optim.SGD(net.parameters(),lr = 0.05)
loss_fn = tc.nn.MSELoss()
for t in range(200):
    prediction = net(x)
    loss = loss_fn(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%5 == 0:
        plt.figure(1, figsize=(10, 3))
        plt.subplot(131)
        plt.cla()
        plt.title('Net1')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.pause(0.1)


# save net and params
save_net(net,'data/net.pkl')
save_net(net,'data/net_params.pkl',entire=False)

# reload net
net2 = resotre_net('data/net.pkl',entire=True)
pred2 = net2(x)
plt.subplot(132)
plt.title('Net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), pred2.data.numpy(), 'r-', lw=5)



# reload params
net3 = resotre_net('data/net_params.pkl',entire=False)
pred3 = net3(x)
plt.subplot(133)
plt.title('Net3')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), pred3.data.numpy(), 'r-', lw=5)
plt.show()