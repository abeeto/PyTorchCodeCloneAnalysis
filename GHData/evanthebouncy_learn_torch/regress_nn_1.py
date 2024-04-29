import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# generate the data
A = np.array([[1.0, 2.0],[3.0, 4.0]])
B = np.array([[4.0, 3.0],[2.0, 1.0]])

def to_torch(x):
  x = Variable(torch.from_numpy(x)).type(torch.cuda.FloatTensor)
  return x

def gen_xy():
  x = np.random.rand(2)
  y = np.matmul(A,x) if np.sum(x) > 1.0 else np.matmul(B,x)
  return x, y

def gen_xy_batch():
  xs, ys = [], []
  for i in range(30):
    x,y = gen_xy()
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys)

print (gen_xy())

n_hidden = 200

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(2, n_hidden)
    self.fc2 = nn.Linear(n_hidden, n_hidden)
    self.fc3 = nn.Linear(n_hidden, 2)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-3

net = Net().cuda()
print (net)
params = list(net.parameters())
print(len(params))
print(params[1].size())
optimizer = optim.Adam(net.parameters(), lr=0.01)

for t in range(5000):
  x, y = gen_xy_batch()
  x = to_torch(x)
  y = to_torch(y)

  optimizer.zero_grad()   # zero the gradient buffers
  y_pred = net(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Zero the gradients before running the backward pass.
  loss.backward()
  optimizer.step()



for i in range(100):
  print ("========================")
  x, y = gen_xy()
  print (x)

  print ("prediction ")
  print (net(to_torch(x)))
  print ("truth")
  print (y)
