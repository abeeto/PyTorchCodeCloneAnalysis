import numpy as np
import torch

if __name__ == '__main__':
    arr = np.ndarray(shape=(32, 8732))
    arr = torch.tensor(arr)

    loc = np.ndarray(shape=(32, 8732, 4))
    loc = torch.tensor(loc)

    new = arr.unsqueeze(arr.dim()).expand_as(loc)

    print(arr.shape, loc.shape, new.shape)
