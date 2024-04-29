import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def cifar(path, transform, train=True, type='CIFAR10'):
    if type == 'CIFAR10':
        return torchvision.datasets.CIFAR10(path, train=train,
                                            transform=transform,
                                            download=True)
    elif type == 'CIFAR100':
        return torchvision.datasets.CIFAR100(path, train=True,
                                             transform=transform,
                                             download=True)
    else:
        raise ValueError("Allowed type of CIFAR is CIFAR10 or CIFAR100.")


def transformations(training=True):
    mean_papers, std_papers = [125.3, 123.0, 113.9], [63.0, 62.1, 66.7]
    mean, std = [x / 255 for x in mean_papers], [x / 255 for x in std_papers]

    if training:
        return transforms.Compose([
            transforms.Pad(padding=4),
            transforms.RandomCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])