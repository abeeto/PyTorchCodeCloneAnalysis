
# coding: utf-8

# In[8]:

from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
print(x)


# In[9]:

x = torch.rand(5, 3)
print(x)


# In[12]:

y = torch.rand(5, 3)
y


# In[14]:

result1 =  x + y
result1


# In[25]:

a = torch.ones(5)
a.size()
b = a.numpy()
x = x.cuda()
y = y.cuda()
x+y

