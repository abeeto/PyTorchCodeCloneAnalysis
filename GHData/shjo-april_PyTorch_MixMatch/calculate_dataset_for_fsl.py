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
# for dataset_name in ['MNIST']:

    if dataset_name == 'CIFAR-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR10(data_dir, train_transform=train_transform)

    elif dataset_name == 'CIFAR-100':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_CIFAR100(data_dir, train_transform=train_transform)

    elif dataset_name == 'STL-10':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_STL10(data_dir, train_transform=train_transform)

    elif dataset_name == 'MNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_MNIST(data_dir, train_transform=train_transform)

    elif dataset_name == 'KMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_KMNIST(data_dir, train_transform=train_transform)

    elif dataset_name == 'FashionMNIST':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_FashionMNIST(data_dir, train_transform=train_transform)
    
    elif dataset_name == 'SVHN':
        train_dataset, validation_dataset, test_dataset, in_channels, classes = get_SVHN(data_dir, train_transform=train_transform)
    
    image, label = train_dataset[0]
    print(dataset_name, len(train_dataset), len(validation_dataset), len(test_dataset), image)

'''
CIFAR-10 45000 5000 10000 <PIL.Image.Image image mode=RGB size=32x32 at 0x2CC2B67ACC8>
CIFAR-100 45000 5000 10000 <PIL.Image.Image image mode=RGB size=32x32 at 0x2CC29DFE5C8>
STL-10 4500 500 8000 <PIL.Image.Image image mode=RGB size=96x96 at 0x2CC2BCC7FC8>
MNIST 54004 5996 10000 <PIL.Image.Image image mode=L size=28x28 at 0x2CC2B5BF308>
KMNIST 54000 6000 10000 <PIL.Image.Image image mode=L size=28x28 at 0x2CC2B5BF408>
FashionMNIST 54000 6000 10000 <PIL.Image.Image image mode=L size=28x28 at 0x2CC297BE188>
SVHN 65937 7320 26032 <PIL.Image.Image image mode=RGB size=32x32 at 0x2CC2BCC7D08>
'''