from __future__ import print_function
import numpy as np
import torch

X = torch.empty(5, 3)
print(X)

X = torch.rand(5, 3)
print(X)

X = torch.zeros(5, 3, dtype=torch.long)
print(X)

X = torch.tensor([5.5, 3])
print(X)

X = X.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(X)

X = torch.randn_like(X, dtype=torch.float)    # override dtype!
print(X)                                      # result has the same size

print(X.size())

Y = torch.rand(5, 3)
print(X + Y)

result = torch.empty(5, 3)
torch.add(X, Y, out=result)
print(result)

# adds X to y
Y.add_(X)
print(Y)

print(X[:, 1])

X = torch.randn(4, 4)
Y = X.view(16)
Z = X.view(-1, 8)  # the size -1 is inferred from other dimensions
print(X.size(), Y.size(), Z.size())

X = torch.randn(1)
print(X)
print(X.item())

A = torch.ones(5)
print(A)

B = A.numpy()
print(B)

A.add_(1)
print(A)
print(B)

A = np.ones(5)
B = torch.from_numpy(A)
np.add(A, 1, out=A)
print(A)
print(B)

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    Y = torch.ones_like(X, device=device)  # directly create a tensor on GPU
    # or just use strings ``.to("cuda")``
    X = X.to(device)
    Z = X + Y
    print(Z)
    # ``.to`` can also change dtype together!
    print(Z.to("cpu", torch.double))
