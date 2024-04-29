import torch
 
a = torch.FloatTensor([3,2]) #constructor for 1x2 tensor
print(a)

ca = a.cuda() #converts cpu to gpu
print(ca)

print(a+1)
print(ca+1) #Shows it's cuda, transparent to user
print(ca.device) #Shows it's cuda
