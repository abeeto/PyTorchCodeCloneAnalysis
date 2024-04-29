import torch
from torch.autograd import Variable

# a = torch.tensor([[2,4]],dtype=torch.double,requires_grad=True)
#
# b = torch.zeros(1,2)
#
# b[0,0] = a[0,0]**2 + a[0,1]
# b[0,1] = a[0,1]**3 + a[0,0]
#
# out = 2 * b
# out.backward()
#
# print(out)

a = torch.tensor([2,3])
a *= torch.tensor([4,5])

print(a)