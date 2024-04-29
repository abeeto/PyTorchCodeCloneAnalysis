import os
import torch
import torchvision
from torch.utils.data import DataLoader

def mnist_train_loader(batch_size=5):
    data_dir = './data'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    train_set = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    print('train_set: ', len(train_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader


def mnist_test_loader(batch_size=5):
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    test_set = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    print("test_set ", len(test_set))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return test_loader

