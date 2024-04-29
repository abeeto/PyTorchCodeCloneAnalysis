from torch.utils.data import Dataset
import numpy as np


class MeMnist(Dataset):
    def __init__(self, path, type):
        data = np.load(path)
        self.images = data[type + '_images']
        self.labels = data[type + '_labels']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index, :].flatten(), self.labels[index][0]
