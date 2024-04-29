import numpy as np
import torch

print("pytorch version: {}".format(torch.__version__))
####### NUMPY => TORCH and TORCH => NUMPY ##########
arr = np.array([m for m in range(8)], dtype=np.float32)
print(arr)
print(arr.dtype)
# convert to torch
x = torch.from_numpy(arr)
print(x.dtype)
print(x)
print(x.shape)
# torch tensor refers to np array
arr[0] = 99
print(x)
# for not to share memory :
tensor_ = torch.tensor(arr)
print(tensor_)
arr[1] = 1000
print(tensor_, x)

# seed
torch.manual_seed(42)
print(torch.rand(2, 3))
print(torch.rand(2, 3))
