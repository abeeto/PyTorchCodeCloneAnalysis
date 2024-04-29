# @author: mcaandewiel

import os
import time

import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from PIL import Image

class tMNISTDataset(Dataset):
    def __init__(self, data_file, root_dir, transform=None):
        self.labels = np.load(data_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return int(self.labels.size / 2) - 1
    
    def __getitem__(self, idx):
        [img_name, label] = self.labels[idx + 1]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return (image, int(label))