# from __future__ import print_function
# import torch
#
#
# # x=torch.rand(2,3)
# # #print(x)
# #
# # y=torch.ones(244)
# # #print(y)
# #
# # z=y.numpy();
# # #print(z);
# #
# # print("size of torch vector is "+str(y.size()))
# #
# # print("size of numpy array is"+str(len(z)))
#
#
# #######learning auto grad
#
# x= torch.ones(2,2)
# #print(x)
#
# y=x+2;
# print(y.grad())
from __future__ import print_function
import torch
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
y = x + 2
print(y)
#print(y.grad_fn())
z = y * y * 3
out = z.mean()

print(z, out)
out.backward()
print(x.grad)
