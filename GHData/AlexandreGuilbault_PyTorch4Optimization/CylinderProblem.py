##############################################
# Simple optimization problem using Pytorch

# PROBLEM 8 : A cylindrical can is to hold 20pi m3 
# The material for the top and bottom costs $10/m2 and 
# material for the side costs $8/m2 
# Find the radius r and height h of the most economical can.

# PROBLEM FROM : https://www.math.ucdavis.edu/~kouba/CalcOneDIRECTORY/maxmindirectory/MaxMin.html #8

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

r = Variable(torch.rand(1), requires_grad=True) # radius
opt = torch.optim.Adam([r], lr=0.1)
relu = nn.ReLU()

last_r = -np.inf
while np.abs(r.detach().item() - last_r) > 0.000001:
    last_r = r.detach().item()
    opt.zero_grad()

	#20 = np.pi * r**2 * h
    h = 20/(r**2) # Height

    # Minimize cost
    cost = 2*10*(np.pi*r**2) + 8*(2*np.pi*r*h) #+ 1000*relu(-r) + 1000*relu(-h)
    
    cost.backward()
    opt.step()

    print("r = {:.2f}m , h = {:.2f}m -> Cost : {:.2f}$".format(r.detach().item(), h.detach().item(), cost.detach().item()))
