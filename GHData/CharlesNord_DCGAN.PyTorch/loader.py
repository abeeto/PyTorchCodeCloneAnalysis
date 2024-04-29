""" loader.py
"""
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_loader(config):
    root = os.path.join(os.path.abspath(os.curdir), config.dataset)
    print('[*] Load data from {0}.'.format(root))
    dataset = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(size=config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    return loader

def denorm(x):
    return x * 0.5 + 0.5