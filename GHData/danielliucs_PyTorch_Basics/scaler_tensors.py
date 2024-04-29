import torch

a = torch.tensor([1,2,3]) #Creates a 1x3 tensor with constructor
print(a)

s = a.sum()
print(s) #results in a scaler number

print(s.item()) #.tem() access actual python value of tensor

c = torch.tensor(3) #creates one dimensional tensor with value 3
print(c)
