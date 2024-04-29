import torch

# tensor is just an array - multidimensional array
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
print(x*y)

# specifying shape
x = torch.zeros([2,5])
print(x)
print(x.shape)

# random value tensor generation
y = torch.rand([2,5])
print(y)

# Reshaping for flattening the multidimensional data so it could be fed to the Neural Network
y = y.view([1,10])
print(y)
y = y.view([10,])
print(y)
print(y.shape)


def flatten(t):
    # 1: rows -1: figure out the no. of columns automatically
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t

y = torch.rand([2,5])
print(flatten(y))


# 3D Tensors in a 4D tensor
d4 = torch.rand([2,3,4,2])
print(d4)
print(d4.shape)

# Squeezing, Unsqueezing


# --- Concatenating and Stacking ---
# Axes = Dimensions
# Concat at dim=0 : highest dimension
# d4= 2,3,4,2 after concatenation at axis 0 i.e. highest dimension: d4=1,6,4,2 we have increased the batch size
d6 = torch.cat((d4[0], d4[1]), dim=0)
print(d6.shape)
# d6 = torch.cat((d6[0], d6[1], d6[2], d6[3], d6[4], d6[5]), dim=0) # 24,2
# print(d6.shape)
d6 = torch.cat((d6[0], d6[1], d6[2], d6[3]), dim=1) #

print(d6.shape)