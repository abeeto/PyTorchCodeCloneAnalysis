import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss1 = L1Loss(reduction='sum')
res1 = loss1(inputs, targets)
print(res1)

loss2=MSELoss()
res2=loss2(inputs,targets)
print(res2)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
loss3=CrossEntropyLoss()
res3=loss3(x,y)
print(res3)