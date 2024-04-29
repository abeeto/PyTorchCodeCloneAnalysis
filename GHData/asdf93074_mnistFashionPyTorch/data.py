import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.transforms as transforms

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

trainset = torch.utils.data.DataLoader("fashion-mnist\\data\\fashion\\train-images-idx3-ubyte.gz", batch_size=4, shuffle=True, num_workers=2)
trainlabels = torch.utils.data.DataLoader("fashion-mnist\\data\\fashion\\train-labels-idx1-ubyte.gz", batch_size=4, shuffle=True, num_workers=2)
testset = torch.utils.data.DataLoader("fashion-mnist\\data\\fashion\\t10k-images-idx3-ubyte.gz.gz", batch_size=4, num_workers=2)
testlabels = torch.utils.data.DataLoader("fashion-mnist\\data\\fashion\\t10k-labels-idx1-ubyte.gz", batch_size=4, num_workers=2)

for t in trainset:
    print(t)