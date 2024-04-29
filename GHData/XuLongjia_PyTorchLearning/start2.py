
#本节课学习一下pytroch中的自动求导  pytorch1.0版本已经取消Variable了

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)

#print(tensor)
#print(variable)

#print()
#print(torch.mean(tensor * tensor))

print()
v_out = torch.mean(variable * variable)
v_out.backward()  # x^2/4 求导等于 x/2
print(variable.grad)  #输出导数
print(variable.data)
print(variable.data.numpy())
