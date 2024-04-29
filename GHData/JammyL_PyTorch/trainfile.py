from first_network import * 
import torch
import torchvision
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 
import tqdm as tqdm
import numpy.random as r
import time as time
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))



im_width = 28
im_height = 28


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)

net = Net().to(device)
t1 = time.time()
net.train(trainset, 5)
t2 = time.time()

print(t2 - t1)
torch.save(net, f"Linear-{int(time.time())}.pt")
