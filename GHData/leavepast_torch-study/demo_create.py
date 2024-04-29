import torch
#T为大写
a=torch.Tensor(2,3)
print(a)
#ones为小写
ones=torch.ones(2,3)
print(ones)
#对角线
eye=torch.eye(3,3)
print(eye)
#创建形状一样的tensor
likes=torch.zeros_like(ones)
print(likes)
