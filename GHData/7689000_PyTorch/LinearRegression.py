#%%

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
#%matplotlib inline

torch.__version__
df = pd.read_csv("./mtcars.csv")
df.head()

torch.manual_seed(1234)
x = torch.unsqueeze(torch.from_numpy(np.array(df.qsec)),dim=1)
y = torch.unsqueeze(torch.from_numpy(np.array(df.mpg)),dim=1)

class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
    self.predict = torch.nn.Linear(n_hidden, n_output) # output layer
  def forward(self, x):
    x = F.relu(self.hidden(x)) # activation function for hidden layer
    x = self.predict(x) # linear output
    return x

net = Net(n_feature=1, n_hidden=20, n_output=1)
net.double()
print(net) # Neural network architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss() # this is for regression mean squared loss.
print(optimizer)
print(loss_func)
# turn the interactive mode on
plt.ion()
for t in range(100):
  prediction = net(x) # input x and predict based on x
  loss = loss_func(prediction, y) # must be (1. nn output, 2. target)
  optimizer.zero_grad() # clear gradients for next train
  loss.backward() # backpropagation, compute gradients
  optimizer.step() # apply gradients

  if t % 50 == 0:
    # plot and show learning process
    plt.cla()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'go', lw=3)
    plt.text(0.5, 0, loss.data.numpy())
    plt.show()
plt.ioff()

# %%
