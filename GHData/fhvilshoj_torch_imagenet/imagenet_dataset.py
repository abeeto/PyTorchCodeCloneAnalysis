import os
import pickle
import pathlib
import numpy as np
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

# Use "[folder of this file]/images" as base path
_base_path = (pathlib.Path(__file__).parent.absolute() / 'images').as_posix()

def _get_additional_transforms(transform):
    return transforms.Compose([
        transforms.ToPILImage(),
        transform
    ])

class ImageNetDataset(Dataset):
    def __init__(self, root_dir=_base_path, transform=None):
        self.root_dir   = root_dir
        if self.root_dir[-1] != '/': self.root_dir += '/'


        self.files          = natsorted(os.listdir(self.root_dir))

        # Filter away grayscale images, tif and other funky image formats
        bad_files = os.path.join(root_dir, '../bad_images.pkl')
        if os.path.exists(bad_files):
            with open(bad_files, 'rb') as f:
                bad_images   = set([f.split('/')[-1] for f in pickle.load(f)])
                self.files  = [f for f in self.files if not f in bad_images]

        self.files          = [f for f in self.files if not 'random' in f]
        self.files          = [(os.path.join(self.root_dir, f), int(f.split("_")[0])) for f in self.files]

        self.transform      = _get_additional_transforms(transform)

        with open(self.root_dir + '../imagenet_label_mapping', 'r') as f:
            self.labels = {}
            for l in f:
                num, description = l.split(":")
                self.labels[int(num)] = description.strip()

    def __len__(self):
        return len(self.files)

    def get_label(self, cls):
        if isinstance(cls, torch.Tensor): cls = cls.item()
        return self.labels[cls]

    def __getitem__(self, idx):
        file_name, label = self.files[idx]
        img = np.array(Image.open(file_name))
        
        if self.transform:
            img = self.transform(img)

        return img, label

