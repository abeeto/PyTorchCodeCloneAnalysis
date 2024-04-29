# The structure of training a classification model 
# prepare train and test data: features and label (such as images or texts)
# specify the structure of a neural network model
# train a model
# performance metrics

# Reference Udacity - Deep Learning With PyTorch

import torch
from torchvision import datasets, transforms

# tensor is a main element to be used for calculation in torch
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# https://pytorch.org/docs/stable/tensors.html
ta = torch.from_numpy(np.array([[1,2,3],[7,8,9]]))  
index = torch.tensor([[1,2,1],[0,1,0]])
ta.gather(1, index) # get some items from tensor matrix
ta.unsqueeze(1) # add an additional dimension

# use tensor to calculate gradients
tx = torch.randn(10, requires_grad=True)
y = tx.mean() # y is a scalor
y.backward() # calculate gradients
tx.grad # gradients dy/d(tx) 

# define the structure of a model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# train the model
qnetwork_local = QNetwork(input_size, output_size, seed)
qnetwork_target = QNetwork(input_size, output_size, seed)
optimizer = optim.Adam(qnetwork_local.parameters(), lr=5e-4)
Q_targets_next = qnetwork_target(input).detach().max(1)[0].unsqueeze(1)
Q_targets = rewards + (gamma * Q_targets_next ) # get the model result by getting the max output and use the formula rewards+gamma*max_output
Q_expected = qnetwork_local(input).gather(1, selection) # get the result with a predefined selection
loss = F.mse_loss(Q_expected, Q_targets) # metric is mse
optimizer.zero_grad()
loss.backward() # torch calculate gradients by using backpropgation for the loss function
optimizer.step()

# evaluate the model
qnetwork_local.eval() # use eval to turn off dropout etc before evaluate the model
with torch.no_grad():
  action_values = qnetwork_local(state)
qnetwork_local.train() # use train to get the model ready for training

# get weights
qnetwork_local.state_dict()
