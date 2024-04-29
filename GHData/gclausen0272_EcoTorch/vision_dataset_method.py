from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn

class ImageClassificationDataset(Dataset):
    def __init__(self, num_datapoints=1000, size=(100, 100, 3), num_classes=10, mode='train'):
        self.num_datapoints = num_datapoints
        self.size = np.array(size)
        self.mode = mode
        self.data = torch.randn(self.num_datapoints, 3, 100, 100)
        self.num_classes = num_classes
        self.labels = torch.randint(low=0, high=self.num_classes, size=(1, self.num_datapoints)) 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        labels = self.labels[0][idx]

        return images, labels
