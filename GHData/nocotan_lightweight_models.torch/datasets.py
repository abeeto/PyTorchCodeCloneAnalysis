# -*- coding: utf-8 -*-
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_loaders(dataset="cifar10", root="/tmp/data", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset == "cifar10":
        train_set = CIFAR10(root=root,
                            train=True,
                            transform=transform,
                            download=True)
        test_set = CIFAR10(root=root,
                           train=False,
                           transform=transform,
                           download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        n_classes = 10

    else:
        raise NotImplementedError

    return train_loader, test_loader, n_classes
