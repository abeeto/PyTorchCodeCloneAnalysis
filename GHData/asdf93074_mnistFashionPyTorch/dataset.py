import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import fashionmnist.utils.mnist_reader as mn

class mnistDataSet(Dataset):
    def __init__(self, fname, k, transform=None):
        self.data, self.labels = mn.load_mnist(fname, kind=k)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image,(1,28,28))
        label = self.labels[idx]
        #labels = np.array([0.0 for i in range(10)])
        #labels[label] = 1

        return image, label

    