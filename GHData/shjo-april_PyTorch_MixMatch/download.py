# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from core.data_utils import *
from core.utils import *

data_dir = './data/'
train_transform = transforms.Compose([])

for dataset_name in ['CIFAR-10', 'CIFAR-100', 'STL-10', 'MNIST', 'KMNIST', 'FashionMNIST', 'SVHN']:

    if dataset_name == 'CIFAR-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR10(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'CIFAR-100':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR100(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'STL-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_STL10(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'MNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_MNIST(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'KMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_KMNIST(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'FashionMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_FashionMNIST(data_dir, train_transform=train_transform, download=True)

    elif dataset_name == 'SVHN':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_SVHN(data_dir, train_transform=train_transform, download=True)

    print(dataset_name)
