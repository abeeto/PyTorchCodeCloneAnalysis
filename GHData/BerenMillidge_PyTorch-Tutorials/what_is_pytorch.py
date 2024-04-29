import torch
# okay, so torch seems to be working yay
x = torch.empty(5,3)
print(x)

# okay, so new versoin of python uses python 3 - yay
print(x.size())
# torch operations all follow the same model so it's fine in the end!
# can do array broadcasting or functiosn
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
# you can also provide a tensor as an argument to a function
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
# can also do in place addition operations on tensors
y.add_(x)
# you can use standard numpy like indexing - I do wonder how this i simplemented under the hood
# would be really interesting to look up
# if tensor is one element, you canuse .item() to get it as a python number
x.item()
# you can move tensors to cuda using the .to method - which is funny!