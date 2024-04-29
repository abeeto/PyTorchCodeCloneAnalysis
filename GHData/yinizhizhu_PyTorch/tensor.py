from __future__ import print_function
import torch

x = torch.Tensor(5, 3)

print(torch.is_tensor(x))
#torch.is_storage(obj)
#torch.set_default_tensor_type(t)


#torch.numel(input) -> int
#return the total number of elements in the 'input' Tensor
a = torch.randn(1,2,3,4,5)
print (a)
print(torch.numel(a))

a = torch.zeros(4,4)
print (a)
print(torch.numel(a))

#torch.set_pintoptions(precision=None,threshold=None,edgeitems=None,linewidth=None,profile-None)