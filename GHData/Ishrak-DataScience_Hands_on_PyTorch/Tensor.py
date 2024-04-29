import torch
x=torch.empty(2,3)
print(x)
print(x.size())
print(x.type())
y=torch.tensor([2.5,0.1])
print(y)
print(y.size())
print(y.type())

x1= torch.rand(2,2)
y1=torch.rand(2,2)
z1=x1+y1
print(z1)
# slicing in tensor 
print(z1[:,1])
print(z1[1,1].item())
w1=z1.view(4)
print(w1.size())