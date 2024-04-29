from __future__ import print_function
import torch
import numpy

#torch.eq(input, other, out=None) -> Tensor
a = torch.eq(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
print(a)


#torch.equal(tensor1, tensor2) -> bool

# torch.ge(input, other, out=None) -> Tensor
#>=
a = torch.ge(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
print (a)

#torch.gt(input, other, out=None) -> Tensor
#>


# torch.kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
# torch.le(input, other, out=None) -> Tensor
#<=
# torch.lt(input, other, out=None) -> Tensor
#<

#Max & Min, Take the max as an example
#torch.max(input) -> float
# torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
#torch.max(input, other, out=None) -> Tensor

# torch.ne(input, other, out=None) -> Tensor
#!=

# torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
x = torch.randn(3, 4)
sorted, indices = torch.sort(x)
print (sorted)
print (indices)
sorted, indices = torch.sort(x, 0)
print (sorted)
print (indices)

# torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
#
#
#
#
#
#
#
#
#