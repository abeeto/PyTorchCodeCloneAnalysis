import numpy as np
import torch

Arr = [[5, 6], [7, 9]]

np.array(Arr)

torch.tensor(Arr)

np.ones((2, 2))

torch.ones((2, 2))

torch.rand(2, 2)

# seeds
np.random.seed(534)
np.random.rand(2, 2)

torch.manual_seed(23)
print(torch.rand(2, 2))

# GPU, all seeds are cpu by default
# if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(5)
#    torch.rand(2, 2).cuda()


# numpy conversion to torch
numpyArrayToTorch = np.ones((2, 2))
# npArrayToTorch = np.ones((2, 2), dtype=np.int32) ## dtype to Select variable type
print(numpyArrayToTorch)
print(type(numpyArrayToTorch))
numpyArrayToTorch = torch.from_numpy(numpyArrayToTorch)
print(numpyArrayToTorch)
print(type(numpyArrayToTorch))
# print(type(numpyArrayToTorch.dtype))

# torch conversion to numpy
torchTensorToNumpy = torch.ones(2, 2)
print(torchTensorToNumpy)
print(torchTensorToNumpy.dtype)
torchTensorToNumpy = torchTensorToNumpy.numpy()
print(torchTensorToNumpy)
print(type(torchTensorToNumpy))

# view
A = torch.ones((2, 2))
print(A.view(4))

# element-wise addition
A = torch.ones((2, 2))
B = torch.ones((2, 2))
C = A + B
# C = torch.add(A, B) ## The same as C=A+B in this case
print(C)

# in place addition
print('Old c tensor')
print(C)

C.add_(A)  # Different from C.add because it increments permanently the tensor
# C.add_(A) is equal to C=C+A
# Meanwhile, C.add(A) is equal to C+A
print("New C tensor after in-place addition")
print(C)
print("Back to previous tensor")
C.sub_(A)

X = torch.mul(2, C)  # C.mul(2)
print(X)
X.mul_(C)
X.div_(C)

# mean is calculated by doing the result of adding all the elements (media in italian)
# 0+1+2+3+4+5+6+7+8+9 = 45
# and dividing it by the index number (10)
# (45:10) = 4.5
meanie = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.double)
meanie.size()
meanie.mean(0)
_2DMeanie = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 33, 6, 7, 8, 54]], dtype=torch.double)
_2DMeanie.size()
_2DMeanie.mean(1)
_2DMeanie.mean(0)

# standard deviation
A = torch.tensor(range(0, 10), dtype=torch.double)
A.std()
