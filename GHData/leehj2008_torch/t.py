import torch
import numpy as np
from torch.autograd import Variable
from torch.tensor import   Tensor
X=np.random.rand(20).reshape(10,2)
print X

Z=Variable(Tensor(X[:,0]**2+X[:,1]**3+10))
print "Z=",Z.data