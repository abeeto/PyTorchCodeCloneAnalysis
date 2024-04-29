import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.n_samples = len(self.data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = self.data[index].astype('float32')
        y = self.labels[index].astype('int')
        sample = x, y
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_x, transform=None):
        self.test_x = test_x
        self.transform = transform
        self.n_samples = len(self.test_x)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = self.test_x[index].astype('float32')
        return x


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.as_tensor(np.asarray(targets))


