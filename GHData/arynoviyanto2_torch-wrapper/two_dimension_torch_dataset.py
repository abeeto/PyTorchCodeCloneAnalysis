from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import os
import pandas as pd

class TwoDimensionTorchDataset(Dataset):
    '''
    PyTorch Dataset
    '''

    def __init__(self, df, image_directory_path, target_field, transforms_func):
        self.df = df
        self.image_directory_path = image_directory_path
        self.target_field  = target_field
        self.transforms_func = transforms_func

    def __getitem__(self, index):
        filename = self.df.filename[index]
        label = self.df[self.target_field][index]

        filename_path = os.path.join(self.image_directory_path, filename)

        img = Image.open(filename_path)

        return self.transforms_func(img), label

    def __len__(self):
        return self.df.shape[0]
