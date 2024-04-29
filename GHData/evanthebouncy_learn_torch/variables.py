import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

x = Variable(torch.rand(3,3).type(dtype), requires_grad=True)
y = Variable(torch.rand(3,3).type(dtype), requires_grad=True)
z = (x + y).mean()

print (x)
print (y)
print (z)

z.backward()
print (x.grad)
print (y.grad)

