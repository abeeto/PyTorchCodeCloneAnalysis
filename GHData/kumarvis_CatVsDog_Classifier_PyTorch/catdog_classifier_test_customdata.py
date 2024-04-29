import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CatDogDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.cat_dog_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.cat_dog_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.cat_dog_frame.iloc[idx, 0])

        image = Image.open(img_name)
        if self.transforms is not None:
            image_tensor = self.transforms(image)

        lbl = self.cat_dog_frame.iloc[idx, 1]
        return (image_tensor, lbl)


test_data_csv_path = 'data/test1/cat_dog_lbl.csv'
root_dir = 'data/test1/'

if __name__ == "__main__":
    # Define transforms
    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    num_workers = 4
    custom_cat_dog_frm_csv = CatDogDataset(test_data_csv_path, root_dir, transform)
    cat_dog_dataset_loader = torch.utils.data.DataLoader(dataset=custom_cat_dog_frm_csv,
                                                    batch_size=10, num_workers = num_workers,
                                                    shuffle=False)

    for images, labels in cat_dog_dataset_loader:
        print('Break')