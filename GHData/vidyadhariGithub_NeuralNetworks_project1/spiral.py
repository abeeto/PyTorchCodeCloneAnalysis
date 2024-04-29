# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        #incoming channels are 2
        self.linear1 = torch.nn.Linear(2, num_hid)
        #sigmoid produces single output 
        self.linear2 = torch.nn.Linear(num_hid, 1)

    def forward(self, input):
        #extract x and y from input
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(x*x + y*y)
        r = r.reshape(-1, 1)
        a = torch.atan2(y,x)
        a = a.reshape(-1, 1)
        S = (r,a)
        coordinates = torch.cat(S, dim=1) #concatenate r and a
        temp = self.linear1(coordinates)
        self.hid1 = torch.tanh(temp) #hidden layer activation
        temp = self.linear2(self.hid1)
        output = torch.sigmoid(temp) #output layer activation
        #output = 0*input[:,0] # CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = torch.nn.Linear(2, num_hid)
        self.linear2 = torch.nn.Linear(num_hid, num_hid)
        #sigmoid produces single output 
        self.linear3 = torch.nn.Linear(num_hid, 1)

    def forward(self, input):
        temp = self.linear1(input)
        self.hid1 = torch.tanh(temp) #hidden layer activation
        temp = self.linear2(self.hid1)
        self.hid2 = torch.tanh(temp)
        
        temp = self.linear3(self.hid2)
        output = torch.sigmoid(temp) #output layer activation
        #output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        if layer > 1:
            var = net.hid2[:, node] >= 0
            x = (var).float() #for RawNet having 2 hidden layers
        else:
            var = net.hid1[:, node] >= 0
            x = (var).float()
        plt.pcolormesh(xrange,yrange,x.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
    # INSERT CODE HERE
