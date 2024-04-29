'''
    Regression

'''

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# create data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*(torch.rand(x.size())) # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
x, y = torch.autograd.Variable(x), Variable(y)

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# create neural network
import torch.nn.functional as F # activate functions

class Net(torch.nn.Module): # inherit from torch Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # inherit __init__ functions
        # define every layer
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer output linearly
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer output linearly

    def forward(self, x): # this is the forward in Module
        # forward propagation value, neural network analyze output value
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net structure

"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# training
# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5) # import net profiles, learning rate
loss_func = torch.nn.MSELoss()  # the loss function of predict and real values


plt.ion()   # draw

for t in range(100):
    prediction = net(x)     # feed x to net, output prediction

    loss = loss_func(prediction, y)        # must be (1. nn output, 3. target)

    optimizer.zero_grad()   # clear the previous step parameter values
    loss.backward()         # error bp
    optimizer.step()        # update net parameters

    # continue
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
