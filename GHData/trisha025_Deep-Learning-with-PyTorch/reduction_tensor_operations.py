import torch

t = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype=torch.float32)

#reduction operations
print(t.sum())
print(t.numel())
print(t.sum().numel())
print(t.sum().numel() < t.numel)

#others
print(t.sum())
print(t.prod())
print(t.mean())
print(t.std())

#reduction output that has multiple elements
t1 = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)

print(t1.sum(dim=0)) #across arrays
print(t1.sum(dim=1))    #across elements

#argmax method
t2 = torch.tensor([
    [4,5,3],
    [6,5,2],
    [7,9,7]
], dtype=torch.float32)

print(t2.max())
print(t2.argmax())
