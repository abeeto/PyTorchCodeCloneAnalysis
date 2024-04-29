import torch
from torch.autograd import Variable
import numpy as np

tensor = torch.FloatTensor([[1, 2], [3, 4]])
varibale = Variable(tensor, requires_grad=True)
# print tensor
# print varibale

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(varibale * varibale)
# print(t_out)
# print(v_out)

v_out.backward()
print(varibale.grad)
print(varibale.data)
print(varibale.data.numpy())