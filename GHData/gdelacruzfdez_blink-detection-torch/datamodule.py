import pytorch_lightning as pl
import pandas as pd
import dataloader
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



class VivitDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset_dirs, test_dataset_dirs, train_transform, test_transform, batch_size):
        super().__init__()
        self.train_dataset_dirs = train_dataset_dirs
        self.test_dataset_dirs = test_dataset_dirs
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size

        self.train_dataset = dataloader.BlinkDetectionLSTMDataset(
            self.train_dataset_dirs, self.train_transform)
        self.test_dataset = dataloader.BlinkDetectionLSTMDataset(
            self.test_dataset_dirs, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    #def test_dataloader(self):
    #    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def get_train_dataframe(self):
        return self.train_dataset.getDataframe().copy()

    def get_val_dataframe(self):
        return self.test_dataset.getDataframe().copy()

    def train_weights(self):
        weights = compute_class_weight(class_weight='balanced',
                                       classes=np.unique(self.train_dataset.targets),
                                       y=self.train_dataset.targets)
        return weights
