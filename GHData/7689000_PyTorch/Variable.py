import torch
import torchvision
import numpy as np
from torch.autograd import Variable

a, b = 12, 23
x1 = Variable(torch.randn(a,b), requires_grad = True)
x2 = Variable(torch.randn(a,b), requires_grad = True)
x3 = Variable(torch.randn(a,b), requires_grad = True)
c = x1 * x2
d = c + x3
e = torch.sum(d)
e.backward()
print(e)

# computing the descriptive statistics: mean
print(torch.mean(torch.tensor([10., 10., 13., 10., 34., 45.,
65., 67., 87., 89., 87., 34.])))
# mean across rows and across columns
d = torch.randn(4,5)
print(d)
print(torch.mean(d,dim=0)) # across rows
print(torch.mean(d,dim=1)) # across columns

# median across rows and across columns
d = torch.randn(4,5)
print(d)
print(torch.median(d,dim=0)) # across rows
print(torch.median(d,dim=1)) # across columns

# mode across rows and across columns
d = torch.randn(4,5)
print(d)
print(torch.mode(d))
print(torch.mode(d,dim=0)) # across rows
print(torch.mode(d,dim=1)) # across columns

# standard deviation across rows and across columns
d = torch.randn(4,5)
print(d)
print(torch.std(d))
print(torch.std(d,dim=0)) # across rows
print(torch.std(d,dim=1)) # across columns

# compute variance across rows and across columns
d = torch.randn(4,5)
print(d)
print(torch.var(d))
print(torch.var(d,dim=0))
print(torch.var(d,dim=1))