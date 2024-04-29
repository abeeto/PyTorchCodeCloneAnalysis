import os
import torch
import numpy as np
torch.device("cuda")
print(torch.cuda.is_available())

import torchvision.transforms as tfs

taget = np.random.randint(0,255,size = 6)

img = taget.reshape(2,1,3)
print(img)

taget = tfs.ToTensor()(img)

print(taget)

print(os.getcwd())