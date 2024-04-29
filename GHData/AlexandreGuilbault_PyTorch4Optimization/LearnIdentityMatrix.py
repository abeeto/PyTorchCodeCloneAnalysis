##############################################
# Simple optimization problem using Pytorch
# to learn Identity matrix by minimizing the
# error between mm(XY) and X.

import torch
from torch.autograd import Variable
import numpy as np

x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5), requires_grad=True)

opt = torch.optim.Adam([y], lr=0.001)

actual_loss = np.inf
while actual_loss > 0.01:
    opt.zero_grad()
    
    loss = torch.abs((x.mm(y) - x)).sum()
    
    loss.backward() 
    opt.step()

    actual_loss = loss.detach().item()
    print("Loss : {}".format(actual_loss))

print("\nLearned identity matrix:\n{}".format(y.data.round()))