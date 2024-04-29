import first_network
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

if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = torch.device("cpu")



test = datasets.MNIST('', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

testset = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=True)


net = torch.load("Linear-1591618287.pt").to(device)


net.test(testset)

for data in testset:
	net.test_single(data)
	break
