import torch

x = torch.rand(5, 3) # Initialize with random values
print(x)
y = torch.rand(5, 3)
print(y)
print(torch.add(x, y))
print(x + y)
result = torch.empty(5, 3) # assign the operation result to a variable.
torch.add(x, y, out=result) # Alternatively, all operation methods have an out parameter to store the result.
print(result)
y.add_(x) # in-place addition. Any operation that mutates a tensor in-place is post-fixed with an _.
print(y)  # Same as y = y + x
print(x.size())  # torch.Size([5, 3])
print(y.size())  # torch.Size([5, 3])
print(torch.numel(x)) # 15 - number of elements in x
# reshape tensors:
x = torch.randn(2, 3)            # Size 2x3
print(x)
y = x.view(6)                    # Resize x to size 6
print(y)
z = x.view(-1, 2)                # Size 3x2
print(z)
# Concatenation in the 0 dimension
print(torch.cat((x, x, x), 0))  # torch.cat concatenates the sequences in given dimension
# Stack
print(torch.stack((x, x))) # Concatenates a sequence of tensors along a new!! dimension.
