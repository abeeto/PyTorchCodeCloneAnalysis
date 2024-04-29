import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from typing import List
from torch import Tensor
from torch.utils.data import Dataset


class CarLicensePlateDataset(Dataset):
    def __init__(self, root_directory: str, data_frame: pd.DataFrame, transform_op: transforms = None):
        self.root_dir = root_directory
        self.annotations = data_frame
        self.transform = transform_op

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[item, 0])
        load_image = Image.open(image_path)

        label_dataframe: pd.DataFrame = self.annotations.iloc[item, 1:]
        label_list: List = list(label_dataframe)
        label: Tensor = torch.from_numpy(np.array(label_list)).float()

        # Scaling the labels to the actual pixel value in image
        # label: Tensor = torch.from_numpy(np.array(label_list) * WIDTH).float()

        if self.transform:
            load_image = self.transform(load_image)

        return load_image, label
