import torch
V_data = [1,2,3,4]
V = torch.tensor(V_data)
print (V,type(V))
print (torch.cat([torch.Tensor([1,2,3]),torch.Tensor([4,5,6])],0))
x=torch.randn(2,3,5)
print (x.size())
x.view(5,6)
print (x.view(5,6))
from torch.autograd import Variable