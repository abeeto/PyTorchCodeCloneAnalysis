from genericpath import isfile
from re import I
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.lin1 = nn.Linear(10, 10) # layer 1
        self.lin2 = nn.Linear(10, 10) # layer 2

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)    
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
                num *= i
        return num

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print('Using GPU')

net = MyNet()
#net.to(mps_device)
#print(net)

# NeuroNet for learning to flip (0 => 1, 1 => 0) neuro values

#if os.path.isfile('MySavedNet.pt'):
#    net = torch.load('MySavedNet.pt')

for i in range(100): # epochs
    inp = [1,0,0,0,1,0,0,0,1,0] # input neuron values
    input = Variable(torch.Tensor([inp for _ in range(10)]))
    #input.to(mps_device)

    #print (input)
    output = net(input) # propagate input values thru net
    #print (output)

    out = [0,1,1,1,0,1,1,1,0,0] # define target values for output neurons
    target = Variable(torch.Tensor([out for _ in range(10)]))
  
    # calculate loss of net
    criterion = nn.MSELoss() 
    loss = criterion(output, target) 
    print(f'Epoch : %i' %i, loss) # print loss
    #print(loss)

    net.zero_grad()  # reset gradient after epoch
    loss.backward()  # make back propagation (change weights)
    optimizer = optim.SGD(net.parameters(), lr=0.11) # optimize net

    optimizer.step() 

torch.save(net, 'MySavedNet.pt')



