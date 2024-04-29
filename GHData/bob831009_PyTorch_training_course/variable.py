import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
'''
tensor([[1., 2.],
        [3., 4.]])
'''

print(variable)
'''
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
'''

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
'''
tensor(7.5000)
'''
print(v_out)
'''
tensor(7.5000, grad_fn=<MeanBackward1>)
'''

v_out.backward()


print(variable) # variable
'''
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
'''
print(variable.data) # tensor
'''
tensor([[1., 2.],
        [3., 4.]])
'''
print(variable.data.numpy()) # numpy
'''
[[1. 2.]
 [3. 4.]]
'''