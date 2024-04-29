import torch
import pandas as pd


class Review():
    def __init__(self, path, transforms=None):
        self.df = pd.read_csv(path)
        self.data = self.df['Review']
        self.label = self.df['Rating']
        self.transforms = transforms

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]

        if self.transforms is not None:
            x = self.transforms(x)
        if y < 3:
            y_tensor = -1

        if y > 3:
            y_tensor = 1
        
        if y == 0:
            y_tensor = 0

        return x,y_tensor