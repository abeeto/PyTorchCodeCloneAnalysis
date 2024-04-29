# classes for Twin Spirals classification task

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from tqdm import tqdm # provides lightweight progress bar, but found that doesn't let long process break on Ctrl-C

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.r = list()
        self.a = list()
    # with 16 hidden nodes, converged in 3200 epochs
    # with 8 nodes, converged in 2600 epochs once!
    # with 4 nodes, got stuck at 92% plateau for several thousand epochs, till 20000
    # with 6 nodes, once gave 99.5%, another time 78%
    # with 7 nodes, once converged in <6000 epochs, another time stuck at 83%, 
    # another in 18000, another 4100. Choosing as answer
        self.in_to_hid = nn.Linear(2, num_hid)
        self.hid_to_out = nn.Linear(num_hid,1)

    '''digests data, converts to polar coordinates, and turns to tensor'''
    def forward(self, input):
        self.r = list() # using append(), so should reset lists for every feedforward run
        self.a = list()
        # for x,y in tqdm(input):
        for x,y in input:
            self.r.append(math.sqrt(x*x+y*y))
            self.a.append(math.atan2(y,x))
        paired_input = list()
        for i in range(len(self.r)):
            paired_input.append([self.r[i], self.a[i]])
        x = torch.FloatTensor(paired_input)
        x = torch.tanh(self.in_to_hid(x))
        self.hid1 = x
        x = torch.sigmoid(self.hid_to_out(x))
        # output = 0*input[:,0] # CHANGE CODE HERE
        return x

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.in_to_hid1 = nn.Linear(2,num_hid)
        self.hid1_to_hid2 = nn.Linear(num_hid, num_hid)
        self.hid2_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        x = torch.tanh(self.in_to_hid1(input))
        self.hid1 = x
        x = torch.tanh(self.hid1_to_hid2(x))
        self.hid2 = x
        output = torch.sigmoid(self.hid2_to_out(x))
        return output

class ExoticRawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.in_to_hid1 = nn.Linear(2,num_hid)
        self.hid1_to_hid2 = nn.Linear(num_hid, num_hid)
        self.hid2_to_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        x = torch.relu(self.in_to_hid1(input))
        self.hid1 = x
        x = torch.relu(self.hid1_to_hid2(x))
        self.hid2 = x
        output = torch.sigmoid(self.hid2_to_out(x))
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
        # net.train() # toggle batch norm, dropout back again

        if layer == 1:
            pred = (net.hid1[:,node] >= 0.5).float()
        
        # using elif instead of else to make choosing of layers a deliberate
        # process. e.g. if use a 3 layer custom network later on, should explicitly
        # add code for it, not end up running wrong code in 'else'
        elif layer == 2:
            pred = (net.hid2[:,node]>= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

