#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-14 0:45
# @Author  : NingAnMe <ninganme@qq.com>
import os

import numpy as np
from PIL import Image
from PIL import ImageFile
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torchvision


from path import DATA_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassifierDatasetTest(torch.utils.data.Dataset):
    def __init__(self, images, image_size=256, transform=None):
        super().__init__()
        self.images = images
        self.transform = transform

        if self.transform:
            self.tx = self.transform
        else:
            self.tx = A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
                ToTensorV2(),
            ])

    def get_x(self, img_path: str):
        image = Image.open(img_path)
        image = np.array(image)
        return self.tx(image=image)['image']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.get_x(os.path.join(DATA_PATH, self.images[idx]['img_path']))
        return x


class ClassifierDataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images, labels, image_size=256, transform=None):
        super(ClassifierDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

        if self.transform:
            self.tx = self.transform
        else:
            self.tx = A.Compose([
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Flip(p=1),
                # A.HorizontalFlip(p=0.5),
                A.RandomGamma(p=0.8, gamma_limit=(80, 120), eps=1e-07),
                A.CoarseDropout(p=0.3, max_holes=10, max_height=8, max_width=8,
                                min_holes=5, min_height=8, min_width=8),
                A.RandomResizedCrop(p=0.5, height=image_size, width=image_size, scale=(0.5, 1.0),
                                    ratio=(1.0, 1.0), interpolation=0),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
                ToTensorV2(),
            ])

    def get_x(self, img_path: str):
        image = Image.open(img_path)
        image = np.array(image)
        return self.tx(image=image)['image']

    def get_y(self, label: str):
        label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery',
                      'church', 'commercial-area', 'desert',
                      'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake',
                      'meadow', 'mine',
                      'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway',
                      'residential-area', 'river', 'road',
                      'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant',
                      'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court',
                      'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
        return label_list.index(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.get_x(os.path.join(DATA_PATH, self.images[idx]['img_path']))
        y = self.get_y(self.labels[idx]['label'])
        return x, y
