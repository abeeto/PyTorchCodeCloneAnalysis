# https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671/2

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

num_epochs = 8
num_classes = 10
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_dataset = datasets.FashionMNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

test_dataset = datasets.FashionMNIST(root='./data',
                           train=False,
                           download=True,transform=transform)