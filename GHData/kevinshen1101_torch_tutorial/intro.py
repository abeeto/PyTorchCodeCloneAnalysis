from __future__ import print_function
import torch


# 5 by 3 matrix
x = torch.empty(5, 3)
#print(x)

# 5 by 3 random matrix
x = torch.rand(5, 3)
#print(x)

x = torch.zeros(5, 3, dtype=torch.long)
#print(x)

x = torch.tensor([5.5, 3])
#print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
#print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
#print(x)                                      # result has the same size

#print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

'''
Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
'''

#Numpy indexing
print(x)
print(x[:, 1])

'''
Resizing: If you want to resize/reshape tensor, you can use torch.view:
'''
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

'''
If you have a one element tensor, use .item() to get the value as a Python number
'''
x = torch.randn(1)
print(x)
print(x.item())

