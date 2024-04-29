import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision


# basic information of tensor
a = torch.tensor((1, 2))
print(a.type())
print(a.size())
print(a.dim())


# data type conversion
# Set default tensor type. Float in PyTorch is much faster than double.
torch.set_default_tensor_type(torch.FloatTensor)

a = a.cuda()
print(a)
a = a.cpu()
print(a)
a = a.float()
print(a)
a = a.long()
print(a)


# torch.Tensor np.ndarray
# torch.Tensor -> np.ndarray.
a_array = a.cpu().numpy()

# np.ndarray -> torch.Tensor.
b = np.ones((1, 3))
b = torch.from_numpy(b).float()
b = torch.from_numpy(b.copy()).float()   # If ndarray has negative stride


# tensor PIL.Image
# torch.Tensor -> PIL.Image.
image = PIL.Image.fromarray(torch.clamp(b * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(b)    # Equivalently way

# PIL.Image -> torch.Tensor.
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2, 0, 1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))  # Equivalently way


# np.ndarray PIL.Image
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astype(np.uint8))

# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))


# extract value from tensor which contains one element
value = tensor.item()


# tensor shape change
tensor = torch.reshape(tensor, shape)

# shuffle
tensor = tensor[torch.randperm(tensor.size(0))]  # Shuffle the first dimension

# flip horizontal
# Assume tensor has shape N*D*H*W.
tensor = tensor[:, :, :, torch.arange(tensor.size(3) - 1, -1, -1).long()]

# copy tensor
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                |

# concatenate tensor
tensor = torch.cat(list_of_tensors, dim=0)    # 3 10x5 to 30x5
tensor = torch.stack(list_of_tensors, dim=0)  # 3x10x5


# tranfer interger to one-hot
N = tensor.size(0)
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())


# get non-zero elements
torch.nonzero(tensor)               # Index of non-zero elements
torch.nonzero(tensor == 0)          # Index of zero elements
torch.nonzero(tensor).size(0)       # Number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # Number of zero elements


# tensor expand
a = torch.ones((64, 512))
a = a.view((64, 512, 1, 1)).expand(64, 512, 7, 7)


# matrix multiply
# Matrix multiplication: (m*n) * (n*p) -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p).
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2


# calcute Euclidean distance
# X1 is of shape m*d.
X1 = torch.unsqueeze(X1, dim=1).expand(m, n, d)
# X2 is of shape n*d.
X2 = torch.unsqueeze(X2, dim=0).expand(m, n, d)
# dist is of shape m*n, where dist[i][j] = sqrt(|X1[i, :] - X[j, :]|^2)
dist = torch.sqrt(torch.sum((X1 - X2) ** 2, dim=2))


