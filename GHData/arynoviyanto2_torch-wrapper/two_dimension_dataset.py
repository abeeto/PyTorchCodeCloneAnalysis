# Copyright Ary Noviyanto 2021

from PIL import Image
from sklearn import model_selection
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from two_dimension_torch_dataset import TwoDimensionTorchDataset
from enum import Enum

import torchvision.transforms as transforms
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class DataMode(Enum):
    TRAIN = 'train'
    TEST = 'test'

class TwoDimensionDataset():
    """
    Insert description here

    dataset_directory contains
    /
        images/
        [_created_folder_after_resize]/
        metadata.csv


    """

    train_metadata_filename = '_train.csv'

    def __init__(self, dataset_dir_path, metadata_filename='metadata.csv', active_image_directory = 'images', nFold = 10, frac = 1, target_field='label', custom_image_size=64):
        self.dataset_dir_path = dataset_dir_path
        self.metadata_filename = metadata_filename
        self.active_image_directory = active_image_directory
        self.nFold = nFold
        self.frac = frac
        self.target_field = target_field

        # Load train metadata to DataFrame 
        train_metadata_path = os.path.join(self.dataset_dir_path, self.train_metadata_filename)
        if not os.path.exists(train_metadata_path):
            self.df = pd.read_csv(os.path.join(self.dataset_dir_path, self.metadata_filename))
            self.constructKFoldTrainingPattern()
        else:
            self.df = pd.read_csv(train_metadata_path)
            self.df.reset_index(drop=True)

        # Preprocessing
        if custom_image_size is not None:
            self.active_image_directory = '{active_image_directory}_{custom_image_size}'.format(active_image_directory=self.active_image_directory, custom_image_size=custom_image_size)
            self.squareNormalisation(custom_image_size)

    def constructKFoldTrainingPattern(self):
        self.df['fold'] = -1
        self.df = self.df.sample(frac=self.frac).reset_index(drop=True)
        target = self.df[self.target_field].values

        kf = model_selection.StratifiedKFold(n_splits=self.nFold)
        for fold, (_, test_index) in enumerate(kf.split(X=self.df, y=target)):
            self.df.loc[test_index, 'fold'] = fold

        self.df.to_csv(os.path.join(self.dataset_dir_path, self.train_metadata_filename), index=False)

    def squareNormalisation(self, size=None):
        output_dir = os.path.join(self.dataset_dir_path, self.active_image_directory)
        if not os.path.isdir(output_dir):  
            os.makedirs(output_dir)
        else:
            return
        
        image_file_path_list = [os.path.join(self.dataset_dir_path, 'images', filename) for filename in self.df.filename.values]
        Parallel(n_jobs=4)(
            delayed(self.resize)(image_file_path, size) for image_file_path in tqdm(image_file_path_list)
        )

    def resize(self, image_file_path, size):
        img = Image.open(image_file_path)

        min_size = img.size[0] if img.size[0] < img.size[1] else img.size[1]
        half_min_size = min_size / 2
        left = int(img.size[0] / 2 - half_min_size) # Horizontal
        upper = int(img.size[1] / 2 - half_min_size) # Vertical
        right = left + min_size
        lower = upper + min_size
        cropped_size = size if size is not None else min_size

        cropped_img = img.crop((left, upper, right, lower)).resize((cropped_size, cropped_size), Image.BILINEAR)

        filename = os.path.basename(image_file_path)
        outpath = self.getImageFilePath(filename)

        cropped_img.save(outpath)
        
        # img.thumbnail(size, Image.ANTIALIAS)

    def getImageDirectoryPath(self):
        return os.path.join(self.dataset_dir_path, self.active_image_directory)

    def getImageFilePath(self, filename):
        return os.path.join(self.getImageDirectoryPath(), filename)

    def getTrainTransformsFunc(self):
        return transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def getTestTransformsFunc(self):
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])

    def getDataLoader(self, params):
        if self.nFold != 1: # for validation
            return

        transforms_func = self.getTestTransformsFunc()
        ds = TwoDimensionTorchDataset(self.df, self.getImageDirectoryPath(), self.target_field, transforms_func)
        return DataLoader(ds, batch_size=params['batch_size'], shuffle=False)

    def getDataLoaders(self, params):
        test_dataloaders = {}
        train_dataloaders = {}

        for i in range(self.nFold):
            test_df = self.df.loc[self.df.fold.values == i].reset_index(drop=True)
            test_ds = TwoDimensionTorchDataset(test_df, self.getImageDirectoryPath(), self.target_field, self.getTestTransformsFunc())
            test_dataloaders[i] = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False)


            train_df = self.df.loc[self.df.fold.values != i].reset_index(drop=True)
            train_ds = TwoDimensionTorchDataset(train_df, self.getImageDirectoryPath(), self.target_field, self.getTrainTransformsFunc())
            train_dataloaders[i] = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

        return train_dataloaders, test_dataloaders

    def getTargets(self):
        targets = self.df[self.target_field].unique()
        return targets, len(targets)

    def sample(self, seed_val=888, sample_num=3, randomly=False, row_index_list=[]):
        n = self.df.shape[0]
        if (n < sample_num):
            return

        if len(row_index_list) > 0 :
            sampled_df = self.df.loc[row_index_list, :].reset_index(drop=True)
        else:
            if randomly:
                sampled_df = self.df.sample(sample_num).reset_index(drop=True)
            else:    
                sampled_df = self.df.sample(sample_num, random_state=seed_val).reset_index(drop=True)

        sampled_dataset = TwoDimensionTorchDataset(sampled_df, self.getImageDirectoryPath(), self.target_field, self.getTestTransformsFunc())
        sampled_dataloaders = DataLoader(sampled_dataset, batch_size=sample_num, shuffle=False)
        
        inputs, labels = next(iter(sampled_dataloaders))

        grid_img = torchvision.utils.make_grid(inputs)

        plt.imshow(grid_img.numpy().transpose((1, 2, 0)))
        plt.title([str(i) for i in labels.numpy()])
        plt.show()