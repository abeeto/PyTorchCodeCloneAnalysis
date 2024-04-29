from __future__ import print_function
import torch
import numpy

x = torch.randn(2,3)
print(x)

#torch.cat(seq, dim=0, out=None) -> Tensor
#Concatenates the given sequence of seq Tensors in the given dimension.
print (torch.cat((x,x,x), 0))   #vertical
print (torch.cat((x,x,x), 1))   #horirotal

#torch.chunk(tensor, chunks, dim=0)
#Splits a tensor into a number of chunks along a given dimension.
#torch.gather(input, dim, index, out=None) -> Tensor
t = torch.Tensor([[1,2],[3,4]])
x = torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
print (x)


#torch.index_select(input, dim, index, out=None) -> Tensor
#torch.masked_select(input, mask, out=None) -> Tensor
x = torch.randn(3, 4)
print(x)

indices = torch.LongTensor([0, 2])
print(indices)
y = torch.index_select(x, 0, indices)
print (y)

y = torch.index_select(x, 1, indices)
print(y)


#torch.masked_select(input, mask, out=None) -> Tensor
mask = x.ge(0.5)
print(mask)

mask = torch.masked_select(x, mask)
print(mask)

# torch.nonzero(input, out=None) -> LongTensor
#torch.split(tensor, split_size, dim=0)
#torch.squeeze(input, dim=None, out=None)
#torch.stack(sequence, dim=0, out=None)
# torch.t(input, out=None) -> Tensor
#torch.transpose(input, dim0, dim1, out=None) -> Tensor
#torch.unbind(tensor, dim=0)
# torch.unsqueeze(input, dim, out=None)