#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:27:59
@author      :Caihao (Chris) Cui
@file        :dataset.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib


import torch
import numpy as np

import utils
import torch.utils.data
from skimage.io import imread, imsave

import os
import numpy as np


class DatasetSegmentation(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path):
        #         super(DataLoaderSegmentation, self).__init__()
        self.imgfolder = image_path
        self.maskfolder = label_path
        self.imgs = list(sorted(os.listdir(image_path)))
        self.masks = list(sorted(os.listdir(label_path)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgfolder, self.imgs[idx])
        mask_path = os.path.join(self.maskfolder, self.masks[idx])
        data = imread(img_path)
        data = np.moveaxis(data, -1, 0)
        label = imread(mask_path)
        label = label/255
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

    def __len__(self):
        return len(self.imgs)


def create_image_loader(image_path, label_path, batch_size=16, shuffle=True):
    dataset = DatasetSegmentation(image_path, label_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader


def full_image_loader(image_path, label_path, tile_size):

    dataset = tile_dataset(image_path, label_path, tile_size=tile_size)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    return loader


def training_loader(image_path, label_path, batch_size, tile_size, shuffle=False):

    tile_stride_ratio = 0.5

    dataset = tile_dataset(image_path, label_path, tile_size,
                           tile_stride_ratio=tile_stride_ratio)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # default is 1, use 4 to compare the performance.
        # use memory pining to enable fast data transfer to CUDA-enabled GPU.
        pin_memory=True,
    )

    return loader


def tile_dataset(image_path, label_path, tile_size, tile_stride_ratio=1.0):

    # TODO: Perform data augmentation in this

    x_image = utils.input_image(image_path).convert("RGB")
    y_image = utils.label_image(label_path).convert("1")

    # assert x_image.size == y_image.size

    tile_stride = (
        int(tile_size[0] * tile_stride_ratio),
        int(tile_size[1] * tile_stride_ratio),
    )

    tile_count, extended_size = utils.tiled_image_size(
        x_image.size, tile_size, tile_stride_ratio
    )

    x_extended = utils.extend_image(x_image, extended_size, color=0)
    y_extended = utils.extend_image(y_image, extended_size, color=255)

    x_tiles = np.zeros((tile_count, 3, tile_size[0], tile_size[1]))
    y_tiles = np.zeros((tile_count, tile_size[0], tile_size[1]))

    def tile_generator():
        for x in range(0, extended_size[0], tile_stride[0]):
            for y in range(0, extended_size[1], tile_stride[1]):
                yield (x, y, tile_size[0], tile_size[1])

    for n, (x, y, w, h) in enumerate(tile_generator()):

        box = (x, y, x + w, y + h)

        x_tile = np.array(x_extended.crop(box))
        y_tile = np.array(y_extended.crop(box))

        x_tiles[n, :, :, :] = np.moveaxis(x_tile, -1, 0)
        y_tiles[n, :, :] = y_tile

    # Clip tiles accumulators to the actual number of tiles
    # Since some tiles might have been discarded, n <= tile_count
    x_tiles = torch.from_numpy(x_tiles[0: n + 1, :, :, :])
    y_tiles = torch.from_numpy(y_tiles[0: n + 1, :, :])
    # x_tiles = torch.from_numpy(x_tiles)
    # y_tiles = torch.from_numpy(y_tiles)
    x_tiles = x_tiles.to(dtype=utils.x_dtype())
    y_tiles = y_tiles.to(dtype=utils.y_dtype())

    dataset = torch.utils.data.TensorDataset(x_tiles, y_tiles)

    return dataset
