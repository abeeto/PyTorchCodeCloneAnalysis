
# coding: utf-8

# In[14]:

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)

y = x + 2

z = y * y *3
out = z.mean()
out.backward()
x.grad


# In[20]:

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x*2
while y.data.norm()<1000:
    y = y*2

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

x.grad

