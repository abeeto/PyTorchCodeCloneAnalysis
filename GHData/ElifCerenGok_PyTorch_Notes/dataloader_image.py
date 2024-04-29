import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io

class CatsAndDogs(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    #getitem function will reach the specific image and corresponding target
    def __getitem__(self, index):
        #self.annotations.iloc[index,0] is the image file name in this case
        img_pth = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = io.imread(img_pth)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)







