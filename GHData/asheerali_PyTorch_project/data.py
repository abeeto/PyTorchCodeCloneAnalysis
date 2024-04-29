import torch
import numpy as np
import torchvision as tv
from skimage.io import imread
from skimage.color import gray2rgb
from torch.utils.data import Dataset

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        super().__init__()
        self.data = np.array(data)
        self.mode = mode

        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = gray2rgb(imread(self.data[index][0]))
        labels = np.array(self.data[index][1:], dtype=float)
        return self._transform(images), torch.tensor(labels)
