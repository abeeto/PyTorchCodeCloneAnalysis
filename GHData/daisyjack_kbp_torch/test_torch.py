import torch
from torch.autograd import Variable

a = Variable(torch.LongTensor(2,3), requires_grad=True)
print a
# torch.save(a, 'a.pt')
# b = torch.load('a.pt')
# b.requires_grad = False
# print b.requires_grad
x, y = torch.max(a,1, keepdim = True)
print x, y
import numpy as np
b = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10, 11]]])
print b
print b[:, [0,1], [1,0]]
x = torch.randn(2, 3)
print x
print x.unsqueeze(1).expand(2,4,3)
