import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('./wine.csv', dtype=np.float32,
                delimiter=',', skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        # apply transform to items
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
transforms = torchvision.transforms.Compose([
    ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=ToTensor())
firstitem = dataset[0]
features, labels = firstitem
print(features, labels)


newdataset = WineDataset(transform = transforms)
firstitem = newdataset[0]
features, labels = firstitem
print(features, labels)


newdataset = WineDataset(transform = transforms)

