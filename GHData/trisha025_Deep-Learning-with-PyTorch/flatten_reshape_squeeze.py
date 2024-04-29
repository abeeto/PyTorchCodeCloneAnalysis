import torch

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)

#In PyTorch, two methods are there to access the shape programmatically
print(t.size())  #size method
print(t.shape)   #shape attribute

print(len(t.shape))     #obtaining the tensor rank

#no. of elements in tensor
print(torch.tensor(t.shape).prod())     
print(t.numel())

#reshaping tensor
print(t.reshape(1,12))
print(t.reshape(2,6))
print(t.reshape(3,4))
print(t.reshape(4,3))
print(t.reshape(6,2))
print(t.reshape(12,1))

#squeezing and unsqueezing tensor
print(t.reshape(1,12))
print(t.reshape(1,12).shape)

print(t.reshape(1,12).squeeze)
print(t.reshape(1,12).squeeze().shape)

print(t.reshape(1,12).squeeze().unsqueeze(dim=0))
print(t.reshape(1,12).squeeze().unsqueeze(dim=0).shape)

 #squeezing using flattening function
def flatten(t):
     t = t.reshape(1,-1)
     t = t.squeeze()
     return t

print(flatten(t))

print(t.reshape(1,12))

#print(t.reshape(-1))

#concatenation operations to understand the concept of shape
t1 = torch.tensor([
    [1,2],
    [3,4]
])

t2 = torch.tensor([
    [5,6],
    [7,8]
])

print(torch.cat((t1,t2), dim=0))