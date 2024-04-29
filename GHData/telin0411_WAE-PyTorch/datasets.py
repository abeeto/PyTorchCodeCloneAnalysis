import os
import sys
import random
import logging
import numpy as np
from tqdm import tqdm
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


def data_provider(data_dir, batch_size, img_size=64, norm=0.5, num_workers=1,
                  shuffle=True, isCrop=False, crop_size=140, mode="celebA"):
    assert mode in ["celebA", "CelebA", "cifar10", "CIFAR10", "MNIST", "mnist"]

    if mode in ["celebA", "CelebA"]:
        ## data transformation
        transform_comp_list = []

        # center crop
        if isCrop:
            trans_component = transforms.CenterCrop(crop_size)
            transform_comp_list.append(trans_component)
        # scale to desired image size
        trans_component = transforms.Scale(img_size)
        transform_comp_list.append(trans_component)
        # to tensor
        trans_component = transforms.ToTensor()
        transform_comp_list.append(trans_component)
        # if normalize
        if norm:
            trans_component = transforms.Normalize((norm, norm, norm),
                                                   (norm, norm, norm))
            transform_comp_list.append(trans_component)
        # compose the transformation
        transform = transforms.Compose(transform_comp_list)

        ## dataset
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=int(num_workers))
        dataloader.shape = [int(num) for num in dataset[0][0].size()]
    else:
        raise NotImplementedError

    return dataloader


## Test
if __name__ == "__main__":
    data_dir = "data/CelebA"
    batch_size = 4
    dataloader = data_provider(data_dir, batch_size, isCrop=True)
    m = next(iter(dataloader))
