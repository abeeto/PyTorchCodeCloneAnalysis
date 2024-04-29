import torch
import numpy as np
from torch.autograd import Variable

# regress a vector to the goal vector [1,2,3,4,5]

dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

x = Variable(torch.rand(5).type(dtype), requires_grad=True)
target = Variable(torch.FloatTensor([1,2,3,4,5]).type(dtype), requires_grad=False)

for i in range(100):
  distance = torch.mean(torch.pow((x - target), 2))
  print ("===========================================")
  print ("distance")
  print (distance)
  print ("current x vector")
  print (x)
  distance.backward(retain_graph=True)
  x_grad = x.grad
  print ("gradient")
  print (x_grad)
  x.data.sub_(x_grad.data * 0.1)
  x.grad.data.zero_()

