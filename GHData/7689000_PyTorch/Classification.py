#%%

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
#%matplotlib inline

size = 20
color = 'bo'

torch.manual_seed(1) # reproducible
# sample data preparation
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

# torch need to train on Variable, so convert sample features to Variable
x, y = Variable(x), Variable(y)
plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(), s=100)
plt.show()

class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
    self.out = torch.nn.Linear(n_hidden, n_output) # output layer

  def forward(self, x):
    x = torch.sigmoid(self.hidden(x)) # activation function for hidden layer
    # x = F.sigmoid(self.hidden(x)) # activation function for hidden layer
    x = self.out(x)
    return x

net = Net(n_feature=2, n_hidden=10, n_output=2) # define the network
# net.double()
print(net) # Neural network architecture
# Loss and optimizer
# Softmax is internally computed.
# Set parameters to be updated
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss() # the target label is not an one-hotted.
print(optimizer)
print(loss_func)
# turn the interactive mode on
plt.ion()

for t in range(100):
  out = net(x) # input x and predict based on x
  loss = loss_func(out, y) # must be (1. nn output, 2. target)
  optimizer.zero_grad() # clear gradients for next train
  loss.backward() # backpropagation, compute gradients
  optimizer.step() # apply gradients

  if t % 10 == 0 or t in [3,6]:
    # plot and show learning process
    plt.cla()
    _, prediction = torch.max(F.softmax(out),1)
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:,0],
                x.data.numpy()[:,1],
                c=pred_y, s=100, lw=0)
    accuracy = sum(pred_y == target_y)/200.
    plt.text(1.5,-4, accuracy,
             fontdict={size: 20, color: 'bo'})
    plt.show()
    
plt.ioff()
# %%
