import torch
import numpy as np

from torchvision import transforms

from core.data_utils import *
from utility.utils import *

customized_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_info_list = [
    # ['CIFAR-10', lambda:datasets.CIFAR10('./data/', train=True, download=False, transform=customized_transform)],
    # ['CIFAR-100', lambda:datasets.CIFAR100('./data/', train=True, download=False, transform=customized_transform)],
    # ['STL-10', lambda:datasets.STL10('./data/', split='train', download=False, transform=customized_transform)],
    # ['MNIST', lambda:datasets.MNIST('./data/', train=True, download=False, transform=customized_transform)],
    # ['KMNIST', lambda:datasets.KMNIST('./data/', train=True, download=False, transform=customized_transform)],
    # ['FashionMNIST', lambda:datasets.FashionMNIST('./data/', train=True, download=False, transform=customized_transform)],
    ['SVHN', lambda:datasets.SVHN('./data/SVHN/', split='train', download=False, transform=customized_transform)],
]

log_path = './data/info_mean_and_std.txt'
if os.path.isfile(log_path): 
    os.remove(log_path)

log_func = lambda string='': log_print(string, log_path)
# log_func = lambda string: print(string)

for name, function in dataset_info_list:
    # 1. load dataset
    train_dataset = function()
    
    means = None
    stds = None

    for image, label in train_dataset:
        mean = torch.mean(image, dim=(1, 2)).numpy()
        std = torch.std(image, dim=(1, 2)).numpy()

        if means is None: 
            means = mean
        else:
            means += mean
        
        if stds is None:
            stds = std
        else:
            stds += std
    
    mean = tuple(means / len(train_dataset))
    std = tuple(stds / len(train_dataset))

    log_func(f'# name={name}, mean={mean}, std={std}')
